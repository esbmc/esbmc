"""Copy a class per int/float signature of its constructor calls.

`Box(1)` and `Box(0.5)` give one `__init__` parameter an int at one call and a float at the
other. The converter types the attribute it stores once, so one of the calls is converted
wrongly or refused. When every call passes arguments whose type is int or float on sight, each
signature after the first gets its own copy of the class, which is the same code under another
name. The copy is exact only while nothing can observe the name, so any use of the class other
than a direct module-level call, and any reflection anywhere in the module, keeps the module
unchanged.
"""

import ast
import copy

from preprocessor.binding_sites import binding_sites
from preprocessor.function_value_methods import _binding_count

_REFLECTION = {
    "type", "repr", "str", "format", "vars", "dir", "getattr", "setattr", "hasattr", "delattr",
    "globals", "locals", "eval", "exec", "__import__"
}
_DUNDERS = {
    "__class__", "__name__", "__qualname__", "__dict__", "__module__", "__mro__", "__bases__",
    "__subclasses__"
}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_SAME_KIND_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv)


def _observes_classes(module):
    """Whether the module can read a class's name or identity other than by naming it, or runs
    code when a class is created."""
    return any(
        isinstance(n, ast.Name) and n.id in _REFLECTION or isinstance(n, ast.Attribute)
        and n.attr in _DUNDERS or isinstance(n, ast.JoinedStr) or isinstance(n, ast.ClassDef)
        and n.keywords or isinstance(n, ast.FunctionDef) and n.name == "__init_subclass__"
        for n in ast.walk(module))


def _plain_bases(cls, classes, abc_names):
    """Whether every ancestor of `cls` is `object`, `abc.ABC` or a module class."""
    for base in cls.bases:
        if not isinstance(base, ast.Name):
            return False
        if base.id in classes:
            if not _plain_bases(classes.pop(base.id), classes, abc_names):
                return False
        elif base.id != "object" and base.id not in abc_names:
            return False
    return True


def _is_plain_init(fn):
    args = fn.args
    return not (fn.decorator_list or args.vararg or args.kwarg or args.kwonlyargs
                or args.posonlyargs or args.defaults) and len(args.args) >= 1


def _abc_names(module):
    return {
        a.asname or a.name
        for s in module.body if isinstance(s, ast.ImportFrom) and s.module == "abc" for a in s.names
        if a.name == "ABC"
    }


def _plain_init(cls, module):
    """The `__init__` of `cls` when its body holds only undecorated methods and it derives only
    from `object`, `abc.ABC` and module classes."""
    classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
    if not _plain_bases(cls, classes, _abc_names(module)):
        return None
    init = None
    for stmt in cls.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if not isinstance(stmt, ast.FunctionDef) or stmt.decorator_list:
            return None
        if stmt.name == "__init__":
            init = stmt
    if cls.decorator_list or init is None or not _is_plain_init(init):
        return None
    return init


class _Typer:
    """The int or float type of a module-level expression, when its form alone shows it."""

    def __init__(self, module, sites):
        self.module = module
        self.sites = sites
        self.values = {}
        self.modules = set()
        for stmt in module.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                self.values[stmt.targets[0].id] = stmt.value
            elif isinstance(stmt, ast.Import):
                self.modules.update(a.asname or a.name for a in stmt.names if "." not in a.name)

    def kind(self, node, depth=0):
        if depth > 32:
            return None
        if isinstance(node, ast.Constant):
            return {int: "int", float: "float"}.get(type(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self.kind(node.operand, depth + 1)
        if isinstance(node, ast.BinOp):
            return self._binop_kind(node, depth)
        if isinstance(node, ast.Name):
            value = self.values.get(node.id)
            if value is None or _binding_count(self.module, node.id) != 1:
                return None
            return self.kind(value, depth + 1)
        if isinstance(node, ast.Call) and not node.keywords:
            return self._call_kind(node)
        return None

    def _binop_kind(self, node, depth):
        left = self.kind(node.left, depth + 1)
        right = self.kind(node.right, depth + 1)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div):
            return "float"
        if isinstance(node.op, _SAME_KIND_OPS):
            return "float" if "float" in (left, right) else "int"
        return None

    def _call_kind(self, call):
        func = call.func
        if isinstance(func, ast.Name) and self.sites[func.id] == 0:
            name = func.id
            if name in ("int", "float") and len(call.args) == 1:
                return name
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and func.value.id in self.modules and self.sites[func.value.id] == 1):
            name = func.attr
        else:
            return None
        if not call.args and name in ("nondet_int", "nondet_float"):
            return name[len("nondet_"):]
        return None


def _module_level_nodes(module):
    """The nodes that run at module level, outside every def, lambda and class."""
    stack = list(module.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _constructor_calls(module, cls):
    """The calls of `cls`, when every read of its name is a module-level call of it."""
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    module_nodes = {id(n) for n in _module_level_nodes(module)}
    calls = []
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id == cls.name:
            call = parents.get(id(node))
            if not (isinstance(call, ast.Call) and call.func is node and id(call) in module_nodes):
                return None
            calls.append(call)
    return calls


def _signatures(calls, init, typer):
    """The argument kinds of each call, when every call passes every parameter a known kind."""
    arity = len(init.args.args) - 1
    signatures = []
    for call in calls:
        if call.keywords or len(call.args) != arity:
            return None
        kinds = tuple(typer.kind(a) for a in call.args)
        if None in kinds:
            return None
        signatures.append(kinds)
    return signatures


def _copy_class(module, cls, typer, sites):
    init = _plain_init(cls, module)
    calls = _constructor_calls(module, cls) if init and sites[cls.name] == 1 else None
    signatures = calls and _signatures(calls, init, typer)
    if not signatures or len(set(signatures)) < 2:
        return []
    names = {signatures[0]: cls.name}
    copies = []
    for signature in dict.fromkeys(signatures):
        if signature in names:
            continue
        name = "%s__esbmc_poly_%s" % (cls.name, "_".join(signature))
        if sites[name]:
            return []
        names[signature] = name
        clone = copy.deepcopy(cls)
        clone.name = name
        copies.append(clone)
    for call, signature in zip(calls, signatures):
        call.func.id = names[signature]
    return copies


def copy_classes_per_argument_kinds(module):
    """Give each int/float signature of a class's constructor calls its own copy of the class."""
    if _observes_classes(module):
        return
    sites = binding_sites(module)
    typer = _Typer(module, sites)
    body = []
    for stmt in module.body:
        body.append(stmt)
        if isinstance(stmt, ast.ClassDef):
            body.extend(_copy_class(module, stmt, typer, sites))
    module.body = body
