"""Model `check_type(value, hint)` of the `_sv_verifier` module (#8281).

The module's own `check_type` matches the hint by reflection (`typing.get_origin`,
`get_args`, `collections.abc`), which the converter cannot model. Its meaning
is fixed, so each call is replaced by the test it makes for its hint, and the
module's bodies of `check_type` and `_matches_type` are never converted, as
the converter already does for the module's `nondet_*` functions.
"""

import ast
import builtins
import copy
import importlib.util
import os
from collections import Counter

_MODULE = "_sv_verifier"
_FUNCTIONS = ("check_type", "_matches_type")
_ITERABLES = ("list", "set")


def _bound_names(node):
    """Names `node` binds in its scope."""
    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [a.asname or a.name.split(".")[0] for a in node.names if a.name != "*"]
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    return []


def _unsupported(node, module_name, detail):
    exc = NotImplementedError(f"unsupported: {detail}")
    exc.esbmc_location = (module_name, node.lineno, node.col_offset)
    return exc


def _parse(filename):
    try:
        with open(filename, "r", encoding="utf-8") as src:
            return ast.parse(src.read())
    except (OSError, SyntaxError, ValueError):
        return None


def _class_in(scope, path):
    """Whether `path` names a class of the module `scope` through class
    statements, each the only binding of its name in its scope."""
    for name in path:
        counts = Counter(b for n in _scope_nodes(scope.body) for b in _bound_names(n))
        scope = next((s for s in scope.body if isinstance(s, ast.ClassDef) and s.name == name),
                     None)
        if scope is None or counts[name] != 1:
            return False
    return True


def _scope_nodes(body):
    """The nodes of `body` that belong to its scope, and the defining nodes of
    the functions and classes it contains, but nothing inside them."""
    stack = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                                 ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            stack.extend(ast.iter_child_nodes(node))


def _imports(module):
    """The dotted origin of each name an absolute import binds, and the
    modules imported with `*`."""
    origins, star_modules = {}, set()
    for n in ast.walk(module):
        if isinstance(n, ast.Import):
            origins.update(
                {a.asname or a.name: [a.name]
                 for a in n.names if a.asname or "." not in a.name})
        elif isinstance(n, ast.ImportFrom) and not n.level:
            for a in n.names:
                if a.name == "*":
                    star_modules.add(n.module)
                else:
                    origins[a.asname or a.name] = [n.module, a.name]
    return origins, star_modules


def _is_ellipsis(node):
    return isinstance(node, ast.Constant) and node.value is Ellipsis


class _Lowering:

    def __init__(self, module, module_name):
        self.module = module
        self.module_name = module_name
        self.counts = Counter(b for n in ast.walk(module) for b in _bound_names(n))
        self.origins, self.star_modules = _imports(module)
        self.classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
        self.fresh = 0

    def qualname(self, expr):
        """The dotted path `expr` denotes, from its root's single binding:
        ['M', 'x'] for an import, [''] + names for a class of this module,
        ['builtins', n] for an unbound builtin, else None."""
        path = []
        while isinstance(expr, ast.Attribute):
            path.insert(0, expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        root = expr.id
        if self.counts[root] == 1 and root in self.origins:
            return self.origins[root] + path
        if self.counts[root] == 1 and root in self.classes:
            return ["", root] + path
        if self.counts[root] == 0 and not self.star_modules and hasattr(builtins, root):
            return ["builtins", root] + path
        return None

    def is_class(self, path):
        if path[0] == "builtins":
            return len(path) == 2 and isinstance(getattr(builtins, path[1]), type)
        if path[0] == "":
            return _class_in(self.module, path[1:])
        if path[0] in (_MODULE, "typing"):
            return False
        try:
            spec = importlib.util.find_spec(path[0])
        except (ImportError, ValueError):
            return False
        origin = spec.origin if spec else None
        tree = _parse(origin) if origin and origin.endswith(".py") else None
        return tree is not None and _class_in(tree, path[1:])

    def is_function(self, expr):
        """Whether `expr` denotes check_type or _matches_type of _sv_verifier."""
        if isinstance(expr, ast.Name) and expr.id in _FUNCTIONS and _MODULE in self.star_modules \
                and self.counts[expr.id] == 0:
            raise _unsupported(
                expr, self.module_name, f"'{expr.id}' imported by 'from {_MODULE} import *'; "
                "import it by name")
        root, path = expr, []
        if isinstance(expr, ast.Attribute):
            root, path = expr.value, [expr.attr]
        if isinstance(root, ast.Name) and self.counts[root.id] > 1 \
                and self.origins.get(root.id, []) + path in ([_MODULE, f] for f in _FUNCTIONS):
            raise _unsupported(
                expr, self.module_name, f"'{root.id}' is bound by an import from {_MODULE} "
                "and again elsewhere")
        return self.function_name(expr) is not None

    def function_name(self, expr):
        path = self.qualname(expr)
        return path[1] if path and len(path) == 2 and path[0] == _MODULE \
            and path[1] in _FUNCTIONS else None

    def name(self, role):
        self.fresh += 1
        return f"__esbmc_check_type_{role}{self.fresh}"

    def is_type_of_none(self, hint):
        """Whether `hint` is `type(None)`."""
        if not isinstance(hint, ast.Call) or hint.keywords or len(hint.args) != 1:
            return False
        arg = hint.args[0]
        return self.qualname(hint.func) == ["builtins", "type"] and isinstance(
            arg, ast.Constant) and arg.value is None

    def predicate(self, value, hint):
        """An expression that is true when `value` matches `hint`."""
        if isinstance(hint, ast.Constant) and hint.value is None:
            return ast.Compare(left=copy.deepcopy(value),
                               ops=[ast.Is()],
                               comparators=[ast.Constant(None)])
        if isinstance(hint, ast.BinOp) and isinstance(hint.op, ast.BitOr):
            return ast.BoolOp(
                op=ast.Or(),
                values=[self.predicate(value, hint.left),
                        self.predicate(value, hint.right)])
        if self.is_type_of_none(hint):
            return self.predicate(value, hint.args[0])
        if isinstance(hint, ast.Subscript):
            return self.generic_predicate(value, hint)
        path = self.qualname(hint)
        if path == ["typing", "Any"]:
            return ast.Constant(True)
        if path is not None and self.is_class(path):
            return self.isinstance(value, hint)
        raise self.refuse(hint)

    def generic_predicate(self, value, hint):
        path = self.qualname(hint.value)
        origin = path[1] if path and path[0] == "builtins" and len(path) == 2 else None
        args = hint.slice.elts if isinstance(hint.slice, ast.Tuple) else [hint.slice]
        tests = self.element_tests(value, origin, args)
        if tests is None:
            raise self.refuse(hint)
        return ast.BoolOp(op=ast.And(), values=[self.isinstance(value, hint.value)] + tests)

    def element_tests(self, value, origin, args):
        """The tests on the elements of a `value` of class `origin`, else None
        when `origin[args]` is not a hint this models."""
        if origin in _ITERABLES and len(args) == 1:
            return [self.all_of(value, args[0])]
        if origin == "tuple" and len(args) == 2 and _is_ellipsis(args[1]):
            return [self.all_of(value, args[0])]
        if origin == "tuple" and args and not any(
                isinstance(a, ast.Tuple) or _is_ellipsis(a) for a in args):
            length = ast.Call(func=ast.Name("len", ast.Load()),
                              args=[copy.deepcopy(value)],
                              keywords=[])
            return [
                ast.Compare(left=length, ops=[ast.Eq()], comparators=[ast.Constant(len(args))])
            ] + [
                self.predicate(
                    ast.Subscript(value=copy.deepcopy(value), slice=ast.Constant(i),
                                  ctx=ast.Load()), a) for i, a in enumerate(args)
            ]
        if origin == "dict" and len(args) == 2:
            key, item = self.name("key"), self.name("item")
            items = ast.Call(func=ast.Attribute(value=copy.deepcopy(value),
                                                attr="items",
                                                ctx=ast.Load()),
                             args=[],
                             keywords=[])
            both = ast.BoolOp(op=ast.And(),
                              values=[
                                  self.predicate(ast.Name(key, ast.Load()), args[0]),
                                  self.predicate(ast.Name(item, ast.Load()), args[1])
                              ])
            target = ast.Tuple([ast.Name(key, ast.Store()),
                                ast.Name(item, ast.Store())], ast.Store())
            return [self.all_over(both, target, items)]
        return None

    def all_of(self, value, elem_hint):
        elem = self.name("elem")
        return self.all_over(self.predicate(ast.Name(elem, ast.Load()), elem_hint),
                             ast.Name(elem, ast.Store()), copy.deepcopy(value))

    @staticmethod
    def all_over(test, target, iterable):
        gen = ast.GeneratorExp(
            elt=test,
            generators=[ast.comprehension(target=target, iter=iterable, ifs=[], is_async=0)])
        return ast.Call(func=ast.Name("all", ast.Load()), args=[gen], keywords=[])

    @staticmethod
    def isinstance(value, cls):
        return ast.Call(func=ast.Name("isinstance", ast.Load()),
                        args=[copy.deepcopy(value), copy.deepcopy(cls)],
                        keywords=[])

    def refuse(self, hint):
        return _unsupported(
            hint, self.module_name,
            f"{_MODULE}.check_type with the hint '{ast.unparse(hint)}', which is neither a "
            "class, None, Any, a union of hints, nor list, set, tuple or dict of hints")

    def refuse_use(self, func):
        return _unsupported(
            func, self.module_name, f"this use of {_MODULE}.{self.function_name(func)}; "
            "only a statement check_type(value, hint) is modelled")

    def lower(self, stmt):
        """The statements that replace the call statement `stmt`."""
        call = stmt.value
        if len(call.args) != 2 or call.keywords or any(
                isinstance(a, ast.Starred)
                for a in call.args) or self.function_name(call.func) != "check_type":
            raise self.refuse_use(call.func)
        value, hint = call.args
        prologue = []
        if not isinstance(value, ast.Name):
            temp = self.name("value")
            prologue.append(ast.Assign(targets=[ast.Name(temp, ast.Store())], value=value))
            value = ast.Name(temp, ast.Load())
        message = ast.Constant(f"expected value of type {ast.unparse(hint)}")
        raise_type_error = ast.Raise(
            exc=ast.Call(func=ast.Name("TypeError", ast.Load()), args=[message], keywords=[]))
        check = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=self.predicate(value, hint)),
                       body=[raise_type_error],
                       orelse=[])
        return [ast.fix_missing_locations(ast.copy_location(s, stmt)) for s in prologue + [check]]


class _CallSites(ast.NodeTransformer):

    def __init__(self, lowering):
        self.lowering = lowering

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and self.lowering.is_function(node.value.func):
            return self.lowering.lower(node)
        return self.generic_visit(node)

    def visit_Name(self, node):
        return self.refuse_other_use(node)

    def visit_Attribute(self, node):
        return self.refuse_other_use(node)

    def refuse_other_use(self, node):
        if isinstance(node.ctx, ast.Load) and self.lowering.is_function(node):
            raise self.lowering.refuse_use(node)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Drop the names lowered away: `_sv_verifier` no longer defines them."""
        if node.module != _MODULE or node.level:
            return node
        node.names = [a for a in node.names if a.name not in _FUNCTIONS]
        return node if node.names else ast.copy_location(ast.Pass(), node)


def lower_check_type(module, module_name):
    """Replace each call of `_sv_verifier.check_type` with its test, and drop
    `check_type` and `_matches_type` from `_sv_verifier` itself. A stub left
    in their place would be taken for a user's function of the same name."""
    if os.path.basename(module_name) == f"{_MODULE}.py":
        module.body = [
            s for s in module.body if not (isinstance(s, ast.FunctionDef) and s.name in _FUNCTIONS)
        ]
        return
    _CallSites(_Lowering(module, module_name)).visit(module)
