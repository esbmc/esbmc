"""Lift a def that another def returns to module level (#8970).

`def make(): def inner(): ...; return inner` creates `inner` anew on every call, but a
function value dispatched on needs one module-level function to call. When `inner` captures
nothing from `make`, the one function and the many are the same except for their identity, so
the pass moves it out under a name of its own and `make` returns that.
"""

import ast
import copy
import symtable

from preprocessor.function_value_types import _own_nodes

_PREFIX = "_esbmc_lift_"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _table(module, name):
    """The symbol table of the module-level def `name`."""
    tables = symtable.symtable(ast.unparse(module), "<module>", "exec")
    return next((t for t in tables.get_children() if t.get_name() == name), None)


def _captures(table, inner):
    """Whether `inner`, a def directly in the scope of `table`, reads a variable of it."""
    children = [t for t in table.get_children() if t.get_name() == inner.name]
    return len(children) != 1 or bool(children[0].get_frees())


def _bindings(outer, name):
    """How many times `outer`'s own scope binds `name`."""
    count = 0
    for node in _own_nodes(outer):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            count += 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            count += 2
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count += node.name == name
        elif isinstance(node, ast.alias):
            count += (node.asname or node.name).split(".")[0] == name
    return count + any(a.arg == name for a in ast.walk(outer.args) if isinstance(a, ast.arg))


def _reads_after(outer, index, name):
    """Whether every read of `name` is in the own scope of `outer`, after statement `index`."""
    own = {id(n) for n in _own_nodes(outer)}
    for position, stmt in enumerate(outer.body):
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == name:
                if position <= index or id(node) not in own:
                    return False
    return True


def _plain(inner):
    """A def with no decorator, default or annotation, which no await or yield suspends."""
    args = inner.args
    return not (inner.decorator_list or args.defaults or any(d is not None
                                                             for d in args.kw_defaults)
                or inner.returns or any(a.annotation for a in args.posonlyargs + args.args +
                                        args.kwonlyargs + [args.vararg, args.kwarg] if a) or any(
                                            isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await))
                                            for n in _own_nodes(inner)))


def _liftable(module, outer, index, inner, taken):
    return isinstance(inner, ast.FunctionDef) and not outer.decorator_list and _plain(inner) \
        and _bindings(outer, inner.name) == 1 and _reads_after(outer, index, inner.name) \
        and f"{_PREFIX}{outer.name}_{inner.name}" not in taken \
        and not _captures(_table(module, outer.name), inner)


def _rename_reads(outer, old, new):
    for node in _own_nodes(outer):
        if isinstance(node, ast.Name) and node.id == old:
            node.id = new


def lift_nested_defs(module, targets):
    """Move each `(outer, inner)` of `targets` that captures nothing to module level.

    Returns the names the moved defs now have.
    """
    taken = {n.id
             for n in ast.walk(module) if isinstance(n, ast.Name)
             } | {n.name
                  for n in ast.walk(module) if isinstance(n, _SCOPES[:2] + _SCOPES[3:])}
    lifted, body = set(), []
    for stmt in module.body:
        body.append(stmt)
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for index, inner in list(enumerate(stmt.body)):
            if (stmt.name, getattr(inner, "name", None)) not in targets:
                continue
            if not _liftable(module, stmt, index, inner, taken):
                continue
            new = f"{_PREFIX}{stmt.name}_{inner.name}"
            _rename_reads(stmt, inner.name, new)
            moved = copy.deepcopy(inner)
            moved.name = new
            body.append(moved)
            stmt.body[index] = ast.copy_location(ast.Pass(), inner)
            lifted.add(new)
    module.body = body
    return lifted
