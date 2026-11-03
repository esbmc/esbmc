"""Lower itertools.chain, and itertools.repeat with a count, where a builtin
drains the iterator at once (#8377).

`list(chain(a, b))` becomes `list(list(a) + list(b))` and `sum(repeat(x, n))`
becomes `sum([x] * n)`. A drain runs no code of the program between steps, so
the lazy iterator and the list yield the same elements. A name bound to such a
call and read once, by a drain in the next statement, is replaced there by the
call. Any other use is left as written, which the converter refuses.
"""

import ast
import functools
from collections import Counter
# pylint: disable=too-many-return-statements,too-many-boolean-expressions

_FUNCTIONS = ("chain", "repeat")
_DRAINS = ("list", "tuple", "sorted", "set", "sum", "min", "max")
_SIMPLE_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Assert, ast.Return)
_LOOPING_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp, ast.SetComp,
                   ast.DictComp, ast.GeneratorExp)


def _bound_names(node):
    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        return [node.id]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    return []


def _imports(module):
    """(bound name, function or None for the module) per top-level import
    from itertools."""
    for stmt in module.body:
        if isinstance(stmt, ast.Import):
            yield from ((a.asname or a.name, None) for a in stmt.names if a.name == "itertools")
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "itertools" and not stmt.level:
            yield from ((a.asname or a.name, a.name) for a in stmt.names if a.name in _FUNCTIONS)


def _itertools_names(module, counts):
    """The names bound once, by a top-level import, to the itertools module
    and to its modelled functions."""
    imports = [(name, f) for name, f in _imports(module) if counts[name] == 1]
    return {name for name, f in imports if f is None}, {name: f for name, f in imports if f}


def _function(func, names):
    modules, functions = names
    if isinstance(func, ast.Name):
        return functions.get(func.id)
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id in modules and func.attr in _FUNCTIONS):
        return func.attr
    return None


def _repeat_operands(call):
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    operands = call.args + [k.value for k in call.keywords if k.arg == "times"]
    if len(operands) != 2 or len(call.keywords) > 1:
        return None
    return operands


def _as_list(call, names):
    """The list expression yielding what `call` iterates, or None."""
    if not isinstance(call, ast.Call):
        return None
    kind = _function(call.func, names)
    if kind == "chain":
        if (not call.args or call.keywords or any(isinstance(a, ast.Starred) for a in call.args)):
            return None
        parts = [
            ast.Call(func=ast.Name(id="list", ctx=ast.Load()), args=[a], keywords=[])
            for a in call.args
        ]
        return functools.reduce(lambda left, right: ast.BinOp(left=left, op=ast.Add(), right=right),
                                parts)
    if kind == "repeat":
        operands = _repeat_operands(call)
        if operands is None:
            return None
        return ast.BinOp(left=ast.List(elts=[operands[0]], ctx=ast.Load()),
                         op=ast.Mult(),
                         right=operands[1])
    return None


def _drained(node, counts):
    """The argument a builtin drain call `node` iterates, or None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DRAINS
            and not counts[node.func.id] and len(node.args) == 1 and not node.keywords
            and not isinstance(node.args[0], ast.Starred)):
        return node.args[0]
    return None


def _drain_once(stmt, name, counts):
    """The drain call reading `name` in simple statement `stmt`, run once."""
    if not isinstance(stmt, _SIMPLE_STATEMENTS):
        return None
    stack = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.NamedExpr, ) + _LOOPING_SCOPES):
            return None
        arg = _drained(node, counts)
        if isinstance(arg, ast.Name) and arg.id == name:
            return node
        stack.extend(ast.iter_child_nodes(node))
    return None


def _is_stable(node, rebindable):
    """Whether `node` reads the same value later in the next statement."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id not in rebindable
    if isinstance(node, ast.UnaryOp):
        return _is_stable(node.operand, rebindable)
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_stable(e, rebindable) for e in node.elts)
    return False


def _inline_single_use(block, names, counts, loads, rebindable):
    i = 0
    while i + 1 < len(block):
        stmt = block[i]
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            name = stmt.targets[0].id
            drain = _drain_once(block[i + 1], name, counts)
            if (drain is not None and counts[name] == loads[name] == 1
                    and _as_list(stmt.value, names) is not None
                    and all(_is_stable(a, rebindable) for a in stmt.value.args)
                    and all(_is_stable(k.value, rebindable) for k in stmt.value.keywords)):
                drain.args[0] = stmt.value
                del block[i]
                continue
        i += 1


def _blocks(module):
    for node in ast.walk(module):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield value


def lower_itertools(module):
    """Lower the drained itertools iterators of `module` to lists."""
    counts = Counter(name for n in ast.walk(module) for name in _bound_names(n))
    names = _itertools_names(module, counts)
    if not names[0] and not names[1]:
        return
    loads = Counter(n.id for n in ast.walk(module)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    rebindable = {
        name
        for n in ast.walk(module) if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names
    }
    for block in list(_blocks(module)):
        _inline_single_use(block, names, counts, loads, rebindable)
    for node in ast.walk(module):
        arg = _drained(node, counts)
        lowered = _as_list(arg, names)
        if lowered is not None:
            node.args[0] = ast.fix_missing_locations(ast.copy_location(lowered, arg))
