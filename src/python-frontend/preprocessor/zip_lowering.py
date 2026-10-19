"""Lower `T = dict(zip(A, B))` to a loop (#8283).

`zip` is lowered as a for loop's iterable and inside `list(...)`, but building a
dict from it reached the unsupported-function marker. The assignment becomes an
empty dict filled by a loop over the same `zip`.
"""

# pylint: disable=invalid-name

import ast
import itertools


def _bound_names(module):
    names = set()
    for n in ast.walk(module):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.alias):
            names.add(n.asname or n.name.split(".")[0])
    return names


def _zip_build(stmt):
    """`(target, zip_call)` for `target = dict(zip(a, b))`, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)):
        return None
    call = stmt.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "dict" and len(call.args) == 1 and not call.keywords):
        return None
    inner = call.args[0]
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            and inner.func.id == "zip" and len(inner.args) == 2 and not inner.keywords):
        return None
    target = stmt.targets[0].id
    if any(isinstance(n, ast.Name) and n.id == target for n in ast.walk(inner)):
        return None
    return target, inner


def _loop(stmt, build, counter):
    target, zip_call = build
    n = next(counter)
    names = [f"ESBMC_zip_{n}_key", f"ESBMC_zip_{n}_value"]
    store = ast.Subscript(value=ast.Name(id=target, ctx=ast.Load()),
                          slice=ast.Name(id=names[0], ctx=ast.Load()),
                          ctx=ast.Store())
    body = ast.Assign(targets=[store], value=ast.Name(id=names[1], ctx=ast.Load()))
    init = ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())],
                      value=ast.Dict(keys=[], values=[]))
    loop = ast.For(target=ast.Tuple(elts=[ast.Name(id=v, ctx=ast.Store()) for v in names],
                                    ctx=ast.Store()),
                   iter=zip_call,
                   body=[body],
                   orelse=[])
    for node in (init, loop):
        ast.copy_location(node, stmt)
    return [init, loop]


def lower_zip_builds(module):
    """Rewrites every eligible zip-built dict in `module`."""
    if _bound_names(module) & {"dict", "zip"}:
        return
    counter = itertools.count()
    for node in ast.walk(module):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list) or not stmts or not isinstance(stmts[0], ast.stmt):
                continue
            out = []
            for stmt in stmts:
                build = _zip_build(stmt)
                out.extend(_loop(stmt, build, counter) if build else [stmt])
            setattr(node, field, out)
    ast.fix_missing_locations(module)
