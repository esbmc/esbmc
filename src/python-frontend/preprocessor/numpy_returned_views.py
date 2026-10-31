"""Pass a view returned by a function straight on as the slice it stands for.

The converter binds a slice passed to a function as a view of its source, with
storage, offset, strides and read-only state, but it cannot see through a call:

    def tail(v):
        return v[1::2]

    mutate(tail(a))    ->    mutate(a[1::2])

Only a function that is a single slice of a parameter qualifies, with plain
names or constants as arguments, so the slice reads what the call would have
and nothing runs in it. A call the rewrite does not cover is left unchanged.
"""

import ast
import copy


def _has_plain_signature(function):
    args = function.args
    return not (function.decorator_list or args.vararg or args.kwarg or args.kwonlyargs
                or args.defaults)


def _is_slice_of_params(value, params):
    if (not isinstance(value, ast.Subscript) or not isinstance(value.slice, ast.Slice)
            or not isinstance(value.value, ast.Name)):
        return False
    return not any(
        isinstance(n, ast.Call) or isinstance(n, ast.Name) and n.id not in params
        for n in ast.walk(value))


def _returned_slice(function):
    """The slice `function` returns if that is all it does, or None."""
    if (not _has_plain_signature(function) or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)):
        return None
    params = {a.arg for a in function.args.posonlyargs + function.args.args}
    value = function.body[0].value
    return value if _is_slice_of_params(value, params) else None


def _qualifying_functions(module):
    """The module's functions returning one slice, by name, each bound only once."""
    bound = {}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound[node.name] = bound.get(node.name, 0) + 1
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound[node.id] = bound.get(node.id, 0) + 1
    return {
        s.name: s
        for s in module.body
        if isinstance(s, ast.FunctionDef) and bound[s.name] == 1 and _returned_slice(s) is not None
    }


def _substituted(call, function):
    """`call` to `function` as the slice it returns, or None if it cannot stand for it."""
    params = [a.arg for a in function.args.posonlyargs + function.args.args]
    if (call.keywords or len(call.args) != len(params)
            or not all(isinstance(a, (ast.Name, ast.Constant)) for a in call.args)):
        return None
    arguments = dict(zip(params, call.args))
    result = copy.deepcopy(_returned_slice(function))
    for parent in ast.walk(result):
        for field, child in ast.iter_fields(parent):
            if isinstance(child, ast.Name):
                setattr(parent, field, copy.deepcopy(arguments[child.id]))
    ast.copy_location(result, call)
    return ast.fix_missing_locations(result)


def _imports_numpy(module):
    return any(
        isinstance(s, ast.Import) and any(a.name == "numpy" for a in s.names) for s in module.body)


def _rewrite_arguments(call, returning):
    for index, arg in enumerate(call.args):
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id in returning):
            call.args[index] = _substituted(arg, returning[arg.func.id]) or arg


def lower_numpy_returned_views(module):
    """Rewrite each qualifying call passed as an argument to a function of the module."""
    if not _imports_numpy(module):
        return
    returning = _qualifying_functions(module)
    defined = {s.name for s in module.body if isinstance(s, ast.FunctionDef)}
    for call in [n for n in ast.walk(module) if isinstance(n, ast.Call)]:
        if isinstance(call.func, ast.Name) and call.func.id in defined:
            _rewrite_arguments(call, returning)
