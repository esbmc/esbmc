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


def _plain_call(node, name, nargs):
    """Whether `node` is `name(...)` with `nargs` positional arguments."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            and len(node.args) == nargs and not node.keywords)


def _zip_build(stmt):
    """`(target, zip_call)` for `target = dict(zip(a, b))`, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and _plain_call(stmt.value, "dict", 1)):
        return None
    inner = stmt.value.args[0]
    target = stmt.targets[0].id
    if not _plain_call(inner, "zip", 2) or any(
            isinstance(n, ast.Name) and n.id == target for n in ast.walk(inner)):
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


def _statement_lists(module):
    for node in ast.walk(module):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield node, field, stmts


def lower_zip_builds(module):
    """Rewrites every eligible zip-built dict in `module`."""
    if _bound_names(module) & {"dict", "zip"}:
        return
    counter = itertools.count()
    for node, field, stmts in _statement_lists(module):
        out = []
        for stmt in stmts:
            build = _zip_build(stmt)
            out.extend(_loop(stmt, build, counter) if build else [stmt])
        setattr(node, field, out)
    ast.fix_missing_locations(module)


def _zip_binding(stmt):
    """`(name, zip_call)` for `name = zip(...)` over names and literals, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and _is_zip_call(stmt.value)):
        return None
    if any(
            isinstance(n, (ast.Call, ast.Starred, ast.NamedExpr)) for arg in stmt.value.args
            for n in ast.walk(arg)):
        return None
    return stmt.targets[0].id, stmt.value


def _is_zip_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip"
            and node.args and not node.keywords)


def _mentions(tree, name):
    return sum(
        1 for n in ast.walk(tree)
        if (isinstance(n, ast.Name) and n.id == name) or (isinstance(n, ast.arg) and n.arg == name))


def _reader(stmt, name):
    """The `for` loop or `list(name)` call in `stmt` that reads `name`, else None."""
    if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Name) and stmt.iter.id == name:
        return stmt
    if hasattr(stmt, "body"):
        return None
    scopes = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    if any(isinstance(n, scopes) and _mentions(n, name) for n in ast.walk(stmt)):
        return None
    return next(
        (n for n in ast.walk(stmt)
         if _plain_call(n, "list", 1) and isinstance(n.args[0], ast.Name) and n.args[0].id == name),
        None)


def _inline_zip(stmts, i, module):
    """Moves the zip bound by `stmts[i]` into the one sibling reading it."""
    name, zip_call = _zip_binding(stmts[i])
    j = next((j for j in range(i + 1, len(stmts)) if _mentions(stmts[j], name)), None)
    reader = None if j is None else _reader(stmts[j], name)
    operands = {n.id for arg in zip_call.args for n in ast.walk(arg) if isinstance(n, ast.Name)}
    if (reader is None or _mentions(module, name) != 2
            or _bound_names(ast.Module(body=stmts[i + 1:j + 1], type_ignores=[])) & operands):
        return False
    if isinstance(reader, ast.For):
        reader.iter = zip_call
    else:
        reader.args[0] = zip_call
    return True


def inline_zip_names(module):
    """Reads `z = zip(...)` at its one `list(z)` or `for ... in z` (#8967).

    The iterator is consumed once, so the call moves to its reader, where the
    existing `list(zip(...))` and `for ... in zip(...)` lowerings apply. Its
    operands are names and literals not rebound in between, so they evaluate
    to the same objects.
    """
    if _bound_names(module) & {"list", "zip"}:
        return
    for node, field, stmts in _statement_lists(module):
        kept = [
            s for i, s in enumerate(stmts)
            if not (_zip_binding(s) and _inline_zip(stmts, i, module))
        ]
        setattr(node, field, kept)
