"""Route the builtin open() to the text-file model in models/io.py (#9314).

A call is routed only in a module that never binds `open`, and only for a
valid text mode spelled as a literal, which is decoded here; any other call
keeps the builtin, which is refused as not modelled.

The model is a class of another module. The converter does not report
CPython's TypeError for such an instance in `len()`, in a loop over an
untyped parameter and the like, so a routed file may only be used through
its modelled attributes, a with statement or a for loop; any other use is
refused.
"""

import ast
import codecs

from preprocessor.binding_sites import binding_sites
from preprocessor.rejection import attach_location

_IO_ALIAS = "__esbmc_io"
_METHODS = {"read", "readline", "write", "close"}
_ATTRS = _METHODS | {"name", "closed"}


def _str_constant(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _text_mode(mode):
    """`(readable, writable)` for a valid text mode, else None."""
    if len(set(mode)) != len(mode) or not set(mode) <= set("rwxat+"):
        return None
    if sum(c in "rwxa" for c in mode) != 1:
        return None
    return "r" in mode or "+" in mode, "r" not in mode or "+" in mode


def _known_encoding(node):
    """Whether `node` names a codec; CPython raises LookupError for any other."""
    if not _str_constant(node):
        return False
    try:
        codecs.lookup(node.value)
    except LookupError:
        return False
    return True


def _mode_flags(call, kwargs):
    """`(readable, writable)` for the literal text mode of `call`, else None."""
    if len(call.args) == 2 and "mode" in kwargs:
        return None
    mode = call.args[1] if len(call.args) == 2 else kwargs.get("mode", ast.Constant("r"))
    return _text_mode(mode.value) if _str_constant(mode) else None


def _enclosing_param(name, node, parents):
    """The parameter `name` of the def enclosing `node`, else None."""
    scope = parents.get(id(node))
    while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        scope = parents.get(id(scope))
    if scope is None:
        return None
    return next((a for a in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs
                 if a.arg == name), None)


def _is_str(node, module, parents, sites):
    """Whether `node` is evidently a str. CPython raises TypeError for a path
    of most other types, which the model, taking a str, would not report."""
    if _str_constant(node) or isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return all(_is_str(side, module, parents, sites) for side in (node.left, node.right))
    return isinstance(node, ast.Name) and sites[node.id] == 1 and _is_str_name(
        node, module, parents, sites)


def _is_str_name(node, module, parents, sites):
    """Whether `node`, bound once, is a str parameter or names a str."""
    param = _enclosing_param(node.id, node, parents)
    if param is not None:
        return isinstance(param.annotation, ast.Name) and param.annotation.id == "str"
    binding = next((n for n in ast.walk(module) if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == node.id), None)
    return binding is not None and _is_str(binding.value, module, parents, sites)


def _routed_args(call):
    """The model's arguments for an `open(...)` call it covers, else None."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "open" and 1 <= len(call.args) <= 2):
        return None
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    if not set(kwargs) <= {"mode", "encoding"}:
        return None
    if "encoding" in kwargs and not _known_encoding(kwargs["encoding"]):
        return None
    flags = _mode_flags(call, kwargs)
    if flags is None:
        return None
    return [call.args[0]] + [ast.Constant(flag) for flag in flags]


def _parents(module):
    return {id(child): node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}


def _bound_name(parent, node):
    """The name `parent` binds to `node`, by `with node as name` or
    `name = node`, else None."""
    if isinstance(parent, ast.withitem) and parent.context_expr is node:
        target = parent.optional_vars
    elif isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1:
        target = parent.targets[0]
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def _is_line_loop(parent, node):
    return (isinstance(parent, ast.For) and parent.iter is node
            and isinstance(parent.target, ast.Name) and not parent.orelse)


def _is_modelled_use(node, parents, file_names):
    """Whether `node`, a routed call or a load of a file name, is used only
    as the model covers."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Attribute):
        if parent.attr not in _ATTRS or not isinstance(parent.ctx, ast.Load):
            return False
        call = parents.get(id(parent))
        return parent.attr not in _METHODS or (isinstance(call, ast.Call) and call.func is parent)
    if _is_line_loop(parent, node):
        return True
    if isinstance(parent, ast.withitem) and parent.optional_vars is None:
        return True
    return isinstance(node, ast.Call) and _bound_name(parent, node) in file_names


def _refuse(node, module_name):
    exc = NotImplementedError(
        "a file from open() is supported only through read(), readline(), write(), "
        "close(), .name, .closed, a with statement or a for loop")
    attach_location(exc, node, module_name)
    raise exc


class _LineLoopLowering(ast.NodeTransformer):
    """Rewrite `for line in <file>:` as a loop over `<file>.readline()` that
    stops at "", as CPython's iteration over a text file does. The model class
    has no `__iter__`."""

    def __init__(self, files):
        self.files = files
        self.count = 0

    def visit_For(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        it = node.iter
        if id(it) not in self.files or not _is_line_loop(node, it):
            return node
        prelude = []
        if not isinstance(it, ast.Name):
            self.count += 1
            name = f"__esbmc_file_{self.count}"
            prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=it))
            it = ast.Name(id=name, ctx=ast.Load())
        line = node.target.id
        loop = ast.parse(f"""
while True:
    {line} = {it.id}.readline()
    if {line} == "":
        break
""").body[0]
        loop.body.extend(node.body)
        return [ast.copy_location(stmt, node) for stmt in prelude + [loop]]


def _import_index(body):
    index = 0
    while index < len(body) and (
        (isinstance(body[index], ast.Expr) and _str_constant(body[index].value)) or
        (isinstance(body[index], ast.ImportFrom) and body[index].module == "__future__")):
        index += 1
    return index


def _file_loads(module, calls, parents, sites):
    """The loads of each name bound once to one of `calls`."""
    names = {_bound_name(parents.get(id(call)), call) for call in calls}
    names = {name for name in names if name and sites[name] == 1}
    return names, [
        n for n in ast.walk(module)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names
    ]


def _route_to_model(module, calls):
    for call, args in calls:
        call.func = ast.Attribute(value=ast.Name(id=_IO_ALIAS, ctx=ast.Load()),
                                  attr="_open_text",
                                  ctx=ast.Load())
        call.args, call.keywords = args, []
    module.body.insert(
        _import_index(module.body),
        ast.copy_location(ast.Import(names=[ast.alias(name="io", asname=_IO_ALIAS)]), calls[0][0]))


def lower_builtin_open(module, module_name):
    """Rewrite each covered `open(...)` to `__esbmc_io._open_text(...)`."""
    sites = binding_sites(module)
    if sites["open"]:
        return
    calls = [(n, args) for n in ast.walk(module) if (args := _routed_args(n)) is not None]
    if not calls:
        return
    parents = _parents(module)
    calls = [(call, args) for call, args in calls if _is_str(args[0], module, parents, sites)]
    if not calls:
        return
    call_nodes = [call for call, _ in calls]
    file_names, loads = _file_loads(module, call_nodes, parents, sites)
    for node in call_nodes + loads:
        if not _is_modelled_use(node, parents, file_names):
            _refuse(node, module_name)
    _LineLoopLowering({id(n) for n in call_nodes + loads}).visit(module)
    _route_to_model(module, calls)
    ast.fix_missing_locations(module)
