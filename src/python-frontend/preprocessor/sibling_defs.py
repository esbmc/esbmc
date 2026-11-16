"""Define a function before the adjacent sibling defs that read it (#8960, #9034).

Running an undecorated def whose defaults and annotations call nothing only
binds its name, so the order of adjacent such defs cannot be observed. The
converter resolves a sibling only once its def is converted, so `def g():
return h` followed by `def h()` refused `h`.
"""

import ast


def _binds_only_its_name(stmt):
    return (isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list and not any(
        isinstance(n, ast.Call) for part in (stmt.args, stmt.returns) if part
        for n in ast.walk(part)))


def _reads(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _read_at_def_time(stmt):
    return _reads(stmt.args) | (_reads(stmt.returns) if stmt.returns else set())


def _ordered(run, reads=_reads):
    """`run` with each def after the defs of `run` it reads, or `run` as it is
    when two defs share a name, read each other, or one reads at def time a
    name a def at or after it binds."""
    names = {d.name for d in run}
    if len(names) != len(run):
        return run
    if any(_read_at_def_time(d) & {later.name for later in run[i:]} for i, d in enumerate(run)):
        return run
    ordered = []
    pending = list(run)
    while pending:
        later = names - {d.name for d in ordered}
        ready = next((d for d in pending if not reads(d) & (later - {d.name})), None)
        if ready is None:
            return run
        ordered.append(ready)
        pending.remove(ready)
    return ordered


def _held_methods(cls):
    """`attr -> names` of the methods that `cls` stores as `obj.attr = obj.m`."""
    held = {}
    for node in ast.walk(cls):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.value, ast.Attribute)):
            held.setdefault(node.targets[0].attr, set()).add(node.value.attr)
    return held


def _method_reads(held):
    """The methods a method reads through an attribute that holds them, which
    the converter calls through only once their defs are converted (#8370)."""

    def reads(method):
        return {
            m
            for n in ast.walk(method)
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
            for m in held.get(n.attr, ())
        }

    return reads


def _reordered(body, reads=_reads):
    result, run = [], []
    for stmt in body:
        if _binds_only_its_name(stmt):
            run.append(stmt)
            continue
        result += _ordered(run, reads) + [stmt]
        run = []
    return result + _ordered(run, reads)


def define_read_siblings_first(module):
    for node in ast.walk(module):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = _reordered(node.body)
        elif isinstance(node, ast.ClassDef):
            node.body = _reordered(node.body, _method_reads(_held_methods(node)))
