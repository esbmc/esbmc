"""Drop a module-level class that is rebound before anything can observe it."""

import ast

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _def_time_nodes(func):
    """What a def evaluates when it runs: its body only runs when called."""
    args = func.args
    yield from func.decorator_list
    yield from args.defaults
    yield from (d for d in args.kw_defaults if d is not None)
    yield from (a.annotation for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
                if a.annotation)
    if func.returns:
        yield func.returns


def _executed_nodes(stmt):
    """The nodes evaluated when `stmt` runs."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, _DEFS):
            stack.extend(_def_time_nodes(node))
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _observes(stmt, name):
    """Whether running `stmt` may read `name`: directly, or through a call."""
    return any(
        isinstance(n, ast.Call) or (
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load))
        for n in _executed_nodes(stmt))


def _rebinds(stmt, name):
    if isinstance(stmt, (*_DEFS, ast.ClassDef)):
        return stmt.name == name
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value is not None and isinstance(stmt.target, ast.Name) \
            and stmt.target.id == name
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return any((a.asname or a.name.split(".")[0]) == name for a in stmt.names)
    return False


def drop_shadowed_classes(module):
    """Remove each top-level `class C` that a later top-level statement rebinds
    before any statement from the class on could read `C` (#8343). The
    converter resolves a call by class name before any rebinding, so `C = f`
    left `C(...)` constructing the class."""
    body = module.body
    keep = []
    for i, stmt in enumerate(body):
        if isinstance(stmt, ast.ClassDef) and not stmt.decorator_list:
            rest = body[i:]
            j = next((k for k, s in enumerate(rest) if k and _rebinds(s, stmt.name)), None)
            if j is not None and not any(_observes(s, stmt.name) for s in rest[:j + 1]):
                continue
        keep.append(stmt)
    module.body = keep
