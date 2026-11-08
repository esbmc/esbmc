"""Apply a decorator the program itself defines (#8419).

`@d def f` binds `f` to `d(f)`. The converter otherwise ignored a decorator,
so the undecorated function ran instead of what the decorator returns.
"""

import ast
import copy

from preprocessor.check_type_lowering import _bound_names
from preprocessor.rejection import attach_location

UNDECORATED_PREFIX = "ESBMC_undecorated_"


def _decorator_root(decorator):
    return decorator.func if isinstance(decorator, ast.Call) else decorator


def _is_user_decorator(decorator, user_functions):
    root = _decorator_root(decorator)
    return isinstance(root, ast.Name) and root.id in user_functions


def _refuse(node, what, module_name):
    exc = NotImplementedError(f"{what} is not supported")
    attach_location(exc, node, module_name)
    raise exc


def _returns_lambda(function):
    return any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Lambda) for n in ast.walk(function))


def _is_factory_call(decorator):
    """`@factory(args)` whose arguments call nothing: they are evaluated after
    the def here, while CPython evaluates them before its defaults."""
    return isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and not any(
        isinstance(n, ast.Call) for arg in decorator.args + decorator.keywords
        for n in ast.walk(arg))


def _check_decorators(function, defined, original, module_name):
    for decorator in function.decorator_list:
        spelled = ast.unparse(decorator)
        if not (isinstance(decorator, ast.Name) or _is_factory_call(decorator)):
            _refuse(decorator, f"the decorator @{spelled}", module_name)
        if not _is_user_decorator(decorator, defined):
            _refuse(decorator, f"the decorator @{spelled} beside one this program defines",
                    module_name)
        root = _decorator_root(decorator).id
        if any(_is_user_decorator(d, defined) for d in original[id(defined[root])]):
            _refuse(decorator, f"the decorator @{spelled}, itself decorated,", module_name)
        if _returns_lambda(defined[root]):
            _refuse(decorator, f"the decorator @{spelled}, which returns a lambda,", module_name)


def _returned_wrapper(decorator):
    """The def that the body of `decorator` only defines and returns, or None."""
    body = decorator.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 2:
        return None
    wrapper, ret = body
    if not isinstance(wrapper, ast.FunctionDef) or wrapper.decorator_list:
        return None
    returns_it = isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
    return wrapper if returns_it and ret.value.id == wrapper.name else None


def _liftable(wrapper, param):
    """Whether `wrapper` means the same at top level once `param` is replaced:
    it neither binds `param` nor refers to itself or its enclosing scope."""
    nodes = list(ast.walk(wrapper))
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in nodes):
        return False
    if any(isinstance(n, ast.Name) and n.id == wrapper.name for n in nodes):
        return False
    bound = {b for n in nodes[1:] for b in _bound_names(n)}
    return not {param, wrapper.name} & bound


def _has_plain_params(function):
    args = function.args
    return not (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg)


def _lifted_wrapper(decorator, inner, name, captured=None):
    """The wrapper `decorator` returns, as a top-level def `name` that calls
    `inner` in place of the decorator's parameter, or None. `captured` renames
    the names of an enclosing factory the wrapper reads.

    Applying such a decorator has no other effect, so a call to the decorated
    function becomes a direct call, whose arguments type the wrapper's
    parameters (#9319).
    """
    captured = captured or {}
    if len(decorator.args.args) != 1 or not _has_plain_params(decorator):
        return None
    wrapper = _returned_wrapper(decorator)
    if wrapper is None:
        return None
    renames = {**captured, decorator.args.args[0].arg: inner}
    if len(renames) != len(captured) + 1 or not all(_liftable(wrapper, p) for p in renames):
        return None
    if any(isinstance(n, ast.Name) and n.id == decorator.name for n in ast.walk(wrapper)):
        return None
    lifted = copy.deepcopy(wrapper)
    lifted.name = name
    for n in ast.walk(lifted):
        if isinstance(n, ast.Name) and n.id in renames:
            n.id = renames[n.id]
    return lifted


def _factory_arguments(factory, call):
    """Each parameter of `factory` with the expression `call` binds it to, or
    None when the binding is not a plain one."""
    params = [a.arg for a in factory.args.args]
    if not _has_plain_params(factory) or len(call.args) > len(params):
        return None
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    bound = dict(zip(params, call.args))
    for keyword in call.keywords:
        if keyword.arg not in params or keyword.arg in bound:
            return None
        bound[keyword.arg] = keyword.value
    defaults = factory.args.defaults
    for param, default in zip(params[len(params) - len(defaults):], defaults):
        # A default is evaluated once, when the factory is defined.
        if param not in bound and isinstance(default, ast.Constant):
            bound[param] = default
    return bound if len(bound) == len(params) else None


def _lifted_factory(factory, call, inner, name, prefix):
    """`@factory(args)` as an assignment of each argument to a top-level name,
    then the wrapper lifted as `_lifted_wrapper` does, reading those names in
    place of the factory's parameters; or None."""
    bound = _factory_arguments(factory, call)
    decorator = _returned_wrapper(factory)
    if bound is None or decorator is None or not _has_plain_params(decorator):
        return None
    captured = {param: f"{prefix}_{param}" for param in bound}
    lifted = _lifted_wrapper(decorator, inner, name, captured)
    if lifted is None:
        return None
    assignments = [
        ast.Assign(targets=[ast.Name(id=captured[p], ctx=ast.Store())], value=copy.deepcopy(value))
        for p, value in bound.items()
    ]
    return assignments + [lifted]


def _lifted(decorator, definition, value, name, clone_name):
    """The statements lifting `decorator` applied to `value`, or None."""
    if not isinstance(value, ast.Name):
        return None
    if isinstance(decorator, ast.Name):
        lifted = _lifted_wrapper(definition, value.id, name)
        return lifted and [lifted]
    return _lifted_factory(definition, decorator, value.id, name, clone_name)


def _applied_clone(decorator, definition, value, clone_name):
    """A copy of the decorator `definition`, and the call applying it to `value`."""
    clone = copy.deepcopy(definition)
    clone.name = clone_name
    applied = ast.Name(id=clone.name, ctx=ast.Load())
    if isinstance(decorator, ast.Call):
        applied = ast.Call(func=applied,
                           args=copy.deepcopy(decorator.args),
                           keywords=copy.deepcopy(decorator.keywords))
    return clone, ast.Call(func=applied, args=[value], keywords=[])


def _rebind(function, defined, original, module_name):
    """`def f` renamed, then `f = d1(d2(renamed))`, as Python applies them;
    a factory `@d(a)` applies `d(a)`.

    Each application calls its own copy of the decorator. A nested function
    is one static function here, so the function a wrapper captures would
    otherwise be shared by every function the decorator is applied to.
    """
    _check_decorators(function, defined, original, module_name)
    name = function.name
    statements = []
    value = ast.Name(id=UNDECORATED_PREFIX + name, ctx=ast.Load())
    for position, decorator in reversed(list(enumerate(function.decorator_list))):
        definition = defined[_decorator_root(decorator).id]
        clone_name = f"ESBMC_{definition.name}_{position}_for_{name}"
        lifted = _lifted(decorator, definition, value, clone_name if position else name, clone_name)
        if lifted:
            statements.extend(lifted)
            value = ast.Name(id=lifted[-1].name, ctx=ast.Load())
            continue
        clone, value = _applied_clone(decorator, definition, value, clone_name)
        statements.append(clone)
    function.name = UNDECORATED_PREFIX + name
    function.decorator_list = []
    if not (isinstance(value, ast.Name) and value.id == name):
        statements.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
    for statement in statements:
        ast.fix_missing_locations(ast.copy_location(statement, function))
    return statements


def _refuse_later_decorators(function, user_functions, defined, module_name):
    """Refuse a decorator this program defines only further down."""
    for decorator in function.decorator_list:
        if _is_user_decorator(decorator,
                              user_functions) and _decorator_root(decorator).id not in defined:
            _refuse(decorator, f"the decorator @{ast.unparse(decorator)}, defined after its use,",
                    module_name)


def _refuse_local_decorators(module, module_name):
    """Refuse a decorator a function body defines: it would be ignored."""
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    functions = [
        n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for func in functions:
        local = {s.name for s in func.body if isinstance(s, ast.FunctionDef)}
        for node in ast.walk(func):
            decorators = node.decorator_list if isinstance(node, scopes) else []
            for decorator in decorators:
                if node is not func and _is_user_decorator(decorator, local):
                    _refuse(decorator, "a decorator a function body defines", module_name)


def _refuse_nested_decorators(module, user_functions, module_name):
    """Refuse a decorator this program defines anywhere but on a top-level def."""
    top_level = {id(s) for s in module.body if isinstance(s, ast.FunctionDef)}
    for node in ast.walk(module):
        if not isinstance(
                node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or id(node) in top_level:
            continue
        for decorator in node.decorator_list:
            if _is_user_decorator(decorator, user_functions):
                _refuse(decorator,
                        "a decorator this program defines on a method, nested function or class",
                        module_name)


def lower_user_decorators(module, module_name):
    """Rebind each top-level function decorated by a top-level function of this
    module, using the definition of the decorator that precedes it. Such a
    decorator anywhere else, or defined only further down, is refused, since it
    would be ignored."""
    user_functions = {s.name: s for s in module.body if isinstance(s, ast.FunctionDef)}
    original = {
        id(s): list(s.decorator_list)
        for s in module.body if isinstance(s, ast.FunctionDef)
    }
    defined = {}
    body = []
    for stmt in module.body:
        body.append(stmt)
        if not isinstance(stmt, ast.FunctionDef):
            continue
        name = stmt.name
        _refuse_later_decorators(stmt, user_functions, defined, module_name)
        if any(_is_user_decorator(d, user_functions) for d in stmt.decorator_list):
            body.extend(_rebind(stmt, defined, original, module_name))
        defined[name] = stmt
    module.body = body
    _refuse_nested_decorators(module, user_functions, module_name)
    _refuse_local_decorators(module, module_name)
