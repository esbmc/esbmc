"""Copy a function that a `from` import binds into the importing module (#10342).

`from helper import call` with `def call(h): return h()` in helper.py calls a function value of
the program inside another module, where the dispatch pass does not see it. When `call` is a
plain def of that name in helper's source, reads no name of its module and is only called here,
a copy of it behaves the same in this module. This pass makes the copy, on a copy of the module,
under a name of its own that the calls here then use. The import is left as it is.
"""

import ast
import builtins
import copy

from preprocessor.function_value_methods import _binding_count
from preprocessor.function_value_params import _call_sites
from preprocessor.function_value_types import _Analysis

COPIED = "_esbmc_ci_"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp,
           ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Global, ast.Nonlocal)


def _parameters(fn):
    """Every parameter of `fn`."""
    args = fn.args
    params = args.posonlyargs + args.args + args.kwonlyargs
    return params + [a for a in (args.vararg, args.kwarg) if a is not None]


def _self_contained(fn):
    """Whether `fn` holds no nested scope or default and reads only its own names and
    builtins."""
    body = list(ast.walk(ast.Module(fn.body, [])))
    if fn.decorator_list or fn.args.defaults or any(isinstance(n, _SCOPES) for n in body):
        return False
    if any(d is not None for d in fn.args.kw_defaults):
        return False
    bound = {p.arg for p in _parameters(fn)}
    bound |= {n.id for n in body if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    read = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return not read - bound - set(dir(builtins))


def _copyable(module, analysis, stmt, alias):
    """The def that `alias` of the `from` import `stmt` binds and the calls of it here, when it
    can be copied, else None."""
    path = analysis.local_module(stmt.module or "")
    if path is None or alias.asname is not None or alias.name == "*":
        return None
    fn = analysis.module_def(path, alias.name)
    if fn is None or _binding_count(analysis.loaded[path], alias.name) != 1:
        return None
    if _binding_count(module, alias.name) != 1:
        return None
    sites = _call_sites(module, alias.name)
    if sites is None or any(call.keywords for call in sites[0]):
        return None
    return (fn, sites[0]) if _self_contained(fn) else None


def _renamed_parameters(fn):
    """`fn` with its parameters, and their reads, renamed apart from this module's names. Every
    call passes its arguments by position, so no caller sees the names."""
    params = _parameters(fn)
    renamed = {p.arg: f"{COPIED}{fn.name}_{p.arg}" for p in params}
    for param in params:
        param.arg = renamed[param.arg]
    for node in ast.walk(ast.Module(fn.body, [])):
        if isinstance(node, ast.Name) and node.id in renamed:
            node.id = renamed[node.id]
    return fn


def _located(fn, stmt):
    """`fn` with every node placed at `stmt`, the import that brings it in."""
    for node in ast.walk(fn):
        if hasattr(node, "lineno"):
            ast.copy_location(node, stmt)
    return fn


def copy_imported_functions(module, path):
    """Copy in place each function a top-level `from` import binds that can be; return whether
    any was."""
    analysis = _Analysis(module, path)
    copied = False
    for stmt in list(module.body):
        if not (isinstance(stmt, ast.ImportFrom) and stmt.level == 0):
            continue
        for alias in stmt.names:
            found = _copyable(module, analysis, stmt, alias)
            if found is None:
                continue
            fn, calls = found
            fn = _renamed_parameters(copy.deepcopy(fn))
            fn.name = COPIED + alias.name
            for call in calls:
                call.func.id = fn.name
            module.body.insert(module.body.index(stmt) + 1, _located(fn, stmt))
            copied = True
    if copied:
        ast.fix_missing_locations(module)
    return copied
