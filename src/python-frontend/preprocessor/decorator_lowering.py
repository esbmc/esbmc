"""Apply a decorator the program itself defines (#8419).

`@d def f` binds `f` to `d(f)`. The converter otherwise ignored a decorator,
so the undecorated function ran instead of what the decorator returns.
"""

import ast
import copy

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


def _check_decorators(function, defined, original, module_name):
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Name):
            _refuse(decorator, f"the decorator @{ast.unparse(decorator)}", module_name)
        if not _is_user_decorator(decorator, defined):
            _refuse(decorator, f"the decorator @{decorator.id} beside one this program defines",
                    module_name)
        if any(_is_user_decorator(d, defined) for d in original[id(defined[decorator.id])]):
            _refuse(decorator, f"the decorator @{decorator.id}, itself decorated,", module_name)
        if _returns_lambda(defined[decorator.id]):
            _refuse(decorator, f"the decorator @{decorator.id}, which returns a lambda,",
                    module_name)


def _rebind(function, defined, original, module_name):
    """`def f` renamed, then `f = d1(d2(renamed))`, as Python applies them.

    Each application calls its own copy of the decorator. A nested function
    is one static function here, so the function a wrapper captures would
    otherwise be shared by every function the decorator is applied to.
    """
    _check_decorators(function, defined, original, module_name)
    name = function.name
    statements = []
    value = ast.Name(id=UNDECORATED_PREFIX + name, ctx=ast.Load())
    for position, decorator in reversed(list(enumerate(function.decorator_list))):
        clone = copy.deepcopy(defined[decorator.id])
        clone.name = f"ESBMC_{decorator.id}_{position}_for_{name}"
        statements.append(clone)
        value = ast.Call(func=ast.Name(id=clone.name, ctx=ast.Load()), args=[value], keywords=[])
    function.name = UNDECORATED_PREFIX + name
    function.decorator_list = []
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
