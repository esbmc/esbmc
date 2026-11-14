"""Bind a function parameter that every call passes the same module-level function (#8957).

`def run(f, x): return f(x)` called only as `run(g, ...)` calls `g` through `f`. The dispatch
pass follows a call through a function value only when the value is a dispatched variable, so a
call through such a parameter is made direct: the parameter is dropped, its reads become `g`
and each call omits the argument. Anything that cannot be shown to bind one function keeps the
module unchanged.
"""

import ast
import copy

from preprocessor.binding_sites import binding_sites
from preprocessor.function_value_methods import _binding_count, _is_local_to, _module_scope_walk
from preprocessor.function_value_types import _own_nodes

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _is_plain(fn):
    args = fn.args
    return not (fn.decorator_list or args.vararg or args.kwarg or args.kwonlyargs
                or args.posonlyargs or args.defaults)


def _call_sites(module, name):
    """The calls whose callee is the module-level `name`, or None when it is read otherwise."""
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    calls = []
    for node in _module_scope_walk(module, name):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            call = parents.get(id(node))
            if not (isinstance(call, ast.Call) and call.func is node):
                return None
            calls.append(call)
    return calls, parents


def _enclosing_functions(node, parents):
    parent = parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield parent
        parent = parents.get(id(parent))


def _bound_function(call, index, module, parents):
    """The name of the module-level def that `call` passes at `index`, else None."""
    arg = call.args[index]
    if not (isinstance(arg, ast.Name) and _binding_count(module, arg.id) == 1):
        return None
    if not any(isinstance(s, ast.FunctionDef) and s.name == arg.id for s in module.body):
        return None
    if any(_is_local_to(fn, arg.id) for fn in _enclosing_functions(call, parents)):
        return None
    return arg.id


def _body_keeps(fn, param):
    """The parameter is only read in `fn`, which has no nested scope and does not call itself."""
    nodes = list(_own_nodes(fn))
    if any(isinstance(n, _SCOPES + (ast.Global, ast.Nonlocal)) for n in nodes):
        return False
    return not any(
        isinstance(n, ast.Name) and
        (n.id == fn.name or (n.id == param and not isinstance(n.ctx, ast.Load))) for n in nodes)


def _has_call(nodes):
    return any(isinstance(n, ast.Call) for node in nodes if node for n in ast.walk(node))


def _runs_code(stmt):
    """Whether running `stmt`, at module or class level, can call a function of the module."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return False
    if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign)):
        return not isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _has_call(stmt.decorator_list + [stmt.args, stmt.returns])
    if isinstance(stmt, ast.ClassDef):
        return (_has_call(stmt.decorator_list + stmt.bases + stmt.keywords)
                or any(_runs_code(s) for s in stmt.body))
    return True


def _passed_function(module, fn, index):
    """The one module-level def named at `index` by every call of `fn`, with those calls."""
    sites = _call_sites(module, fn.name)
    if not sites or not sites[0] or _binding_count(module, fn.name) != 1:
        return None
    calls, parents = sites
    if any(c.keywords or any(isinstance(a, ast.Starred)
                             for a in c.args) or len(c.args) != len(fn.args.args) for c in calls):
        return None
    targets = {_bound_function(c, index, module, parents) for c in calls}
    if len(targets) != 1 or None in targets or not _body_keeps(fn, fn.args.args[index].arg):
        return None
    target = targets.pop()
    if target == fn.name or _is_local_to(fn, target):
        return None
    return target, calls


def _specialise(module, fn, index):
    """Drop parameter `index` of `fn` when all its calls pass one module-level function."""
    passed = _passed_function(module, fn, index)
    if passed is None:
        return False
    target, calls = passed
    param = fn.args.args[index].arg
    for node in _own_nodes(fn):
        if isinstance(node, ast.Name) and node.id == param:
            node.id = target
    del fn.args.args[index]
    for call in calls:
        del call.args[index]
    return True


def specialise_function_parameters(module):
    """Make the calls through constant function parameters of module-level defs direct."""
    changed = False
    for fn in [s for s in module.body if isinstance(s, ast.FunctionDef) and _is_plain(s)]:
        for index in reversed(range(len(fn.args.args))):
            changed = _specialise(module, fn, index) or changed
    return changed


def specialise_function_parameters_in_order(module):
    """specialise_function_parameters, when every def precedes the module code that can call it.

    A call made before a def raises NameError in CPython, but would read the function once the
    parameter is dropped (#8369)."""
    ran = False
    for stmt in module.body:
        if ran and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        ran = ran or _runs_code(stmt)
    return specialise_function_parameters(module)


def _def_alias(stmt, sites, defs):
    """The name `stmt` binds when it is `alias = f`, `alias` bound only there and `f` a def of
    `defs`, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)):
        return None
    alias = stmt.targets[0].id
    named = isinstance(stmt.value, ast.Name) and stmt.value.id in defs
    return alias if named and sites[alias] == 1 else None


def _reads_after(module, at, name):
    """The reads of `name` in `module`, or None unless all are at module level after
    `module.body[at]`."""
    reads = [
        n for n in ast.walk(module)
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
    ]
    later = {
        id(n)
        for n in _own_nodes(ast.Module(body=module.body[at + 1:], type_ignores=[]))
        if isinstance(n, ast.Name)
    }
    return reads if all(id(n) in later for n in reads) else None


def inline_def_aliases(module):
    """Read `alias` as `f` where the module binds `alias = f` once, `f` being a def it binds
    once, and reads `alias` only at module level after that binding.

    The analyses of calls through a function value follow calls by the name of a def, so
    `c = f; c(g)` hides from them what `f(g)` shows. Every read of `alias` sees `f`, so the
    program means the same."""
    sites = binding_sites(module)
    defs = {s.name for s in module.body if isinstance(s, ast.FunctionDef) and sites[s.name] == 1}
    for at, stmt in enumerate(module.body):
        alias = _def_alias(stmt, sites, defs)
        for read in (alias and _reads_after(module, at, alias)) or []:
            read.id = stmt.value.id


def _calls_its_parameter(stmt):
    """The name `stmt` binds to a lambda or def whose only effect is to call its one parameter
    with no arguments, as `f = lambda p: p()` or `def f(p): return p()`, else None."""
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Lambda)):
        name, args, result = stmt.targets[0].id, stmt.value.args, stmt.value.body
    elif (isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list and len(stmt.body) == 1
          and isinstance(stmt.body[0], ast.Return)):
        name, args, result = stmt.name, stmt.args, stmt.body[0].value
    else:
        return None
    return name if _calls_only_parameter(args, result) else None


def _calls_only_parameter(args, result):
    """Whether `args` is one plain parameter and `result` calls it with no arguments."""
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults:
        return False
    return (len(args.args) == 1 and isinstance(result, ast.Call)
            and getattr(result.func, "id", None) == args.args[0].arg
            and not (result.args or result.keywords))


def _not_callable_kind(expr, returns):
    """The type name of `expr` when it is a literal, or a call with no arguments of a def in
    `returns`, else None. Calling a value of that type raises TypeError."""
    if isinstance(expr, ast.Constant):
        return type(expr.value).__name__
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and not expr.args
            and not expr.keywords and len(returns.get(expr.func.id, ())) == 1):
        return next(iter(returns[expr.func.id]))
    return None


def _split_call_arms(caller, value, bind, returns):
    """`bind(caller(value))` spelt as an `if` per conditional expression of `value`, with an arm
    that passes a value of a type that cannot be called raising the TypeError of calling it."""
    if isinstance(value, ast.IfExp):
        return [
            ast.If(test=value.test,
                   body=_split_call_arms(caller, value.body, bind, returns),
                   orelse=_split_call_arms(caller, value.orelse, bind, returns))
        ]
    kind = _not_callable_kind(value, returns)
    if kind is None:
        return [bind(ast.Call(ast.Name(caller, ast.Load()), [value], []))]
    error = ast.Raise(
        ast.Call(ast.Name("TypeError", ast.Load()),
                 [ast.Constant(f"'{kind}' object is not callable")], []))
    return ([] if isinstance(value, ast.Constant) else [ast.Expr(value)]) + [error]


def _has_not_callable_arm(value, returns):
    if isinstance(value, ast.IfExp):
        return (_has_not_callable_arm(value.body, returns)
                or _has_not_callable_arm(value.orelse, returns))
    return _not_callable_kind(value, returns) is not None


def raise_not_callable_arms(module):
    """Spell `t = f(a if c else b)` as an `if`, where `f` only calls its one parameter and `b`
    is a value of a type that cannot be called, so that the arm passing `b` raises the TypeError
    CPython raises there and the other arm passes `a` alone.

    Only a statement of the module itself is split, after the one binding of `f`: CPython then
    evaluates `f`, `c`, the arm and the call in the same order."""
    from preprocessor.receiver_split import _return_kinds  # pylint: disable=import-outside-toplevel
    sites = binding_sites(module)
    returns = _return_kinds(module)
    callers = set()
    for at, stmt in enumerate(module.body):
        name = _calls_its_parameter(stmt)
        if name is not None:
            param = (stmt.value.args if isinstance(stmt, ast.Assign) else stmt.args).args[0].arg
            if sites[name] == 1 + (param == name):
                callers.add(name)
            continue
        split = _split_statement(stmt, callers, returns)
        if split is not None:
            module.body[at] = ast.fix_missing_locations(ast.copy_location(split, stmt))


def _split_statement(stmt, callers, returns):
    """`stmt` spelt by `_split_call_arms` when it is `t = f(x)` or `f(x)`, with `f` in `callers`
    and `x` a conditional expression with an arm that cannot be called, else None."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:

        def bind(call):
            return ast.Assign(targets=copy.deepcopy(stmt.targets), value=call)
    elif isinstance(stmt, ast.Expr):
        bind = ast.Expr
    else:
        return None
    call = stmt.value
    if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) in callers
            and len(call.args) == 1 and not call.keywords):
        return None
    if not isinstance(call.args[0], ast.IfExp) or not _has_not_callable_arm(call.args[0], returns):
        return None
    return _split_call_arms(call.func.id, call.args[0], bind, returns)[0]
