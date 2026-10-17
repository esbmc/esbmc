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


def _check_decorators(function, user_functions, module_name):
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Name):
            _refuse(decorator, f"the decorator @{ast.unparse(decorator)}", module_name)
        if not _is_user_decorator(decorator, user_functions):
            _refuse(decorator, f"the decorator @{decorator.id} beside one this program defines",
                    module_name)
        if any(
                _is_user_decorator(d, user_functions)
                for d in user_functions[decorator.id].decorator_list):
            _refuse(decorator, f"the decorator @{decorator.id}, itself decorated,", module_name)


def _rebind(function, user_functions, module_name):
    """`def f` renamed, then `f = d1(d2(renamed))`, as Python applies them.

    Each application calls its own copy of the decorator. A nested function
    is one static function here, so the function a wrapper captures would
    otherwise be shared by every function the decorator is applied to.
    """
    _check_decorators(function, user_functions, module_name)
    name = function.name
    statements = []
    value = ast.Name(id=UNDECORATED_PREFIX + name, ctx=ast.Load())
    for decorator in reversed(function.decorator_list):
        clone = copy.deepcopy(user_functions[decorator.id])
        clone.name = f"ESBMC_{decorator.id}_for_{name}"
        statements.append(clone)
        value = ast.Call(func=ast.Name(id=clone.name, ctx=ast.Load()), args=[value], keywords=[])
    function.name = UNDECORATED_PREFIX + name
    function.decorator_list = []
    statements.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
    for statement in statements:
        ast.fix_missing_locations(ast.copy_location(statement, function))
    return statements


def lower_user_decorators(module, module_name):
    """Rebind each top-level function decorated by a top-level function of this
    module. Such a decorator anywhere else is refused, since it would be
    ignored."""
    user_functions = {s.name: s for s in module.body if isinstance(s, ast.FunctionDef)}
    body = []
    for stmt in module.body:
        body.append(stmt)
        if isinstance(stmt, ast.FunctionDef) and any(
                _is_user_decorator(d, user_functions) for d in stmt.decorator_list):
            body.extend(_rebind(stmt, user_functions, module_name))
    module.body = body
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
