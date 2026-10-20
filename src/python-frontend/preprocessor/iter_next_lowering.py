"""Lower iter() and next() over a list or a user iterator (#8284).

Neither builtin is modelled, so `g = iter(xs)` was refused and `next(g)` was an
unsupported call. Where `g` is bound once and only ever read as `next(g)`, the
iterator becomes an instance of a small per-site class over `list(xs)`, or, for
an instance of a class defining `__iter__` and `__next__`, that instance's own
`__iter__()`; each `next(g)` becomes `g.__next__()`.
"""

# pylint: disable=invalid-name

import ast
from collections import Counter

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _scope_nodes(scope):
    """The nodes of `scope`'s own body, not of the scopes nested in it."""
    stack = list(scope.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _is_call(node, name, nargs):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            and len(node.args) == nargs and not node.keywords)


def _single_assignment(node):
    """`(name, value)` for a statement `name = value`, else `(None, None)`."""
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)):
        return node.targets[0].id, node.value
    return None, None


def _iterator_classes(module):
    """Top-level classes defining both __iter__ and __next__."""
    out = set()
    for stmt in module.body:
        if isinstance(stmt, ast.ClassDef):
            methods = {m.name for m in stmt.body if isinstance(m, ast.FunctionDef)}
            if {"__iter__", "__next__"} <= methods:
                out.add(stmt.name)
    return out


def _list_iterator_class(name):
    return ast.parse(f"""
class {name}:

    def __init__(self, items: list):
        self.items = items
        self.pos = 0

    def __next__(self):
        if self.pos >= len(self.items):
            raise StopIteration
        value = self.items[self.pos]
        self.pos += 1
        return value
""").body[0]


class _Lowering:  # pylint: disable=too-few-public-methods

    def __init__(self, module):
        self.module = module
        bound = Counter()
        for n in ast.walk(module):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                bound[n.id] += 1
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound[n.name] += 1
            elif isinstance(n, ast.arg):
                bound[n.arg] += 1
            elif isinstance(n, ast.alias):
                bound[n.asname or n.name.split(".")[0]] += 1
        self.bound = bound
        self.iterator_classes = _iterator_classes(module)
        self.helpers = []

    def _instances(self, scope):
        """Names bound once in `scope` to a call of an iterator class."""
        return {
            target
            for n in _scope_nodes(scope)
            for target, value in [_single_assignment(n)]
            if target and isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id in self.iterator_classes and self.bound[target] == 1
        }

    def _iterator_source(self, call, instances):
        """What `iter(E)` becomes: E's own __iter__() for an iterator instance,
        else a per-site list iterator over list(E)."""
        arg = call.args[0]
        if isinstance(arg, ast.Name) and arg.id in instances:
            return ast.Call(func=ast.Attribute(value=ast.Name(id=arg.id, ctx=ast.Load()),
                                               attr="__iter__",
                                               ctx=ast.Load()),
                            args=[],
                            keywords=[])
        name = f"ESBMC_ListIterator_{len(self.helpers)}"
        self.helpers.append(_list_iterator_class(name))
        # A list display is already a fresh list; copying it only adds a loop.
        items = arg if isinstance(arg, (ast.List, ast.ListComp)) else ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()), args=[arg], keywords=[])
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[items], keywords=[])

    def lower(self, scope):
        """Lowers every eligible `g = iter(E)` directly in `scope`."""
        nodes = list(_scope_nodes(scope))
        instances = self._instances(scope)
        for stmt in nodes:
            name, value = _single_assignment(stmt)
            if not name or not _is_call(value, "iter", 1):
                continue
            nexts, reads = _uses(nodes, name)
            if self.bound[name] != 1 or not nexts or len(reads) != len(nexts):
                continue
            stmt.value = ast.copy_location(self._iterator_source(stmt.value, instances), stmt.value)
            for call in nexts:
                call.func = ast.Attribute(value=call.args[0], attr="__next__", ctx=ast.Load())
                call.args = []


def _uses(nodes, name):
    """The `next(name)` calls among `nodes`, and every read of `name`."""
    nexts = [
        n for n in nodes
        if _is_call(n, "next", 1) and isinstance(n.args[0], ast.Name) and n.args[0].id == name
    ]
    reads = [
        n for n in nodes if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
    ]
    return nexts, reads


def lower_iter_next(module):
    """Lowers iter()/next() pairs throughout `module` (#8284)."""
    lowering = _Lowering(module)
    if any(lowering.bound[b] for b in ("iter", "next", "list")):
        return
    lowering.lower(module)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lowering.lower(node)
    module.body[0:0] = lowering.helpers
    ast.fix_missing_locations(module)
