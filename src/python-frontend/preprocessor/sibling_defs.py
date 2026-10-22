"""Define a nested function before the adjacent sibling defs that read it (#8960).

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


def _ordered(run):
    """`run` with each def after the defs of `run` it reads, or `run` as it is
    when two defs share a name or read each other."""
    names = {d.name for d in run}
    if len(names) != len(run):
        return run
    ordered = []
    pending = list(run)
    while pending:
        later = names - {d.name for d in ordered}
        ready = next((d for d in pending if not _reads(d) & (later - {d.name})), None)
        if ready is None:
            return run
        ordered.append(ready)
        pending.remove(ready)
    return ordered


def _reordered(body):
    result, run = [], []
    for stmt in body:
        if _binds_only_its_name(stmt):
            run.append(stmt)
            continue
        result += _ordered(run) + [stmt]
        run = []
    return result + _ordered(run)


def define_read_siblings_first(module):
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = _reordered(node.body)
