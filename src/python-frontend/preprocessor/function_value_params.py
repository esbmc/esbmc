"""Bind a function parameter that every call passes the same module-level function (#8957).

`def run(f, x): return f(x)` called only as `run(g, ...)` calls `g` through `f`. The dispatch
pass follows a call through a function value only when the value is a dispatched variable, so a
call through such a parameter is made direct: the parameter is dropped, its reads become `g`
and each call omits the argument. Anything that cannot be shown to bind one function keeps the
module unchanged.
"""

import ast

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
