"""Run a module-level `async def` as a plain function where it is awaited.

With no other task to switch to, `await f(x)` and `asyncio.run(f(x))` run the
body of `f` to completion like `f(x)` would, and a call that is never awaited
evaluates its arguments without running the body. A coroutine used any other
way (stored, passed, scheduled) is refused, as is awaiting anything else.
"""

# pylint: disable=invalid-name,missing-function-docstring

import ast

from preprocessor.rejection import attach_location


def _refuse(node, what, module_name):
    exc = NotImplementedError(f"{what} is not supported")
    attach_location(exc, node, module_name)
    raise exc


def _binding_counts(module):
    counts = {}
    for n in ast.walk(module):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            name = n.id
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = n.name
        elif isinstance(n, ast.arg):
            name = n.arg
        elif isinstance(n, ast.alias):
            name = n.asname or n.name.split(".")[0]
        else:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _is_asyncio_run(call, counts):
    func = call.func
    return (isinstance(func, ast.Attribute) and func.attr == "run"
            and isinstance(func.value, ast.Name) and func.value.id == "asyncio"
            and counts.get("asyncio") == 1 and len(call.args) == 1 and not call.keywords)


def _evaluated_arguments(call, stmt):
    """`call`'s arguments as statements, evaluated as the unawaited call would."""
    values = call.args + [k.value for k in call.keywords]
    return [ast.copy_location(ast.Expr(value=v), stmt)
            for v in values] or [ast.copy_location(ast.Pass(), stmt)]


class _Lowering(ast.NodeTransformer):

    def __init__(self, coroutines, counts, module_name):
        self.coroutines = coroutines
        self.counts = counts
        self.module_name = module_name

    def _coroutine_call(self, node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in self.coroutines)

    def _lower_call(self, call):
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None
                                                                     for k in call.keywords):
            _refuse(call, "unpacking arguments into a coroutine call", self.module_name)
        call.args = [self.visit(a) for a in call.args]
        for k in call.keywords:
            k.value = self.visit(k.value)
        return call

    def visit_Expr(self, node):
        if self._coroutine_call(node.value):
            return _evaluated_arguments(self._lower_call(node.value), node)
        return self.generic_visit(node)

    def visit_Await(self, node):
        if not self._coroutine_call(node.value):
            _refuse(node, "awaiting a value other than a call to a module-level async function",
                    self.module_name)
        return self._lower_call(node.value)

    def visit_Call(self, node):
        if _is_asyncio_run(node, self.counts) and self._coroutine_call(node.args[0]):
            return self._lower_call(node.args[0])
        if self._coroutine_call(node):
            _refuse(node, f"using the coroutine returned by '{node.func.id}' without awaiting it",
                    self.module_name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.coroutines:
            _refuse(node, f"using the async function '{node.id}' as a value", self.module_name)
        return node

    def visit_AsyncFunctionDef(self, node):
        # A nested or method async def is left to the converter as before.
        return node


def _is_generator(fn):
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))


def lower_coroutines(module, module_name):
    """Rewrite each module-level `async def` in `module` as a `def`."""
    defs = [s for s in module.body if isinstance(s, ast.AsyncFunctionDef)]
    if not defs:
        return
    counts = _binding_counts(module)
    coroutines = {
        d.name
        for d in defs if counts[d.name] == 1 and not d.decorator_list and not _is_generator(d)
    }
    lowering = _Lowering(coroutines, counts, module_name)
    for i, stmt in enumerate(module.body):
        if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name in coroutines:
            fn = ast.FunctionDef(**{f: getattr(stmt, f) for f in stmt._fields})
            module.body[i] = ast.copy_location(fn, stmt)
    lowering.visit(module)
    ast.fix_missing_locations(module)
