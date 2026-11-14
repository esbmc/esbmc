"""Call the result of a call that may be a value no call accepts (#10683).

`get()()`, where `get` returns a function on one path and an int on another, raises TypeError
on the int path. The converter gives the result of `get()` one type, so it refuses such a call
or reads the int as a function. When every use of such a value is a call of a call's result,
the value `v` is replaced by `_esbmc_not_callable_of_<n>(v)`, which evaluates `v` and returns a
function of `n` parameters raising TypeError: calling it has the effect calling `v` has.

The value may reach the call through a conditional expression, a name that is only called or
returned, the return of a module-level function whose results are only called, and a
parameter that its function only calls or returns. Any other use keeps the module unchanged.
"""

import ast

from preprocessor.function_value_methods import _binding_count, _is_local_to, _module_scope_walk
from preprocessor.function_value_types import _REFLECTIVE, _REFLECTIVE_ATTRS, _own_nodes

_PREFIX = "_esbmc_not_callable"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp,
           ast.SetComp, ast.DictComp, ast.GeneratorExp)


class _Flow:
    """Where the value of an expression goes, for one module."""

    def __init__(self, module):
        self.module = module
        self.parents = {
            id(child): node
            for node in ast.walk(module)
            for child in ast.iter_child_nodes(node)
        }
        self.seen = set()
        self.counts = {}

    def bindings(self, name):
        if name not in self.counts:
            self.counts[name] = _binding_count(self.module, name)
        return self.counts[name]

    def scope(self, node):
        """The module-level def `node` is in, the module (None) or, in any other scope, False."""
        parent = self.parents.get(id(node))
        while parent is not None and not isinstance(parent, _SCOPES):
            parent = self.parents.get(id(parent))
        if parent is None:
            return None
        return parent if any(
            parent is stmt
            for stmt in self.module.body) and isinstance(parent, ast.FunctionDef) else False

    def sole_def(self, name, at):
        """The undecorated module-level def `name` names where `at` reads it, bound only by it."""
        fn = next(
            (s for s in self.module.body if isinstance(s, ast.FunctionDef) and s.name == name),
            None)
        if fn is None or fn.decorator_list or self.bindings(name) != 1:
            return None
        scope = self.scope(at)
        if scope is False or (scope is not None and _is_local_to(scope, name)):
            return None
        return fn

    def loads(self, name, scope):
        """The reads of `name` bound in `scope` (None for the module), or None when it is also
        updated in place."""
        nodes = _module_scope_walk(self.module, name) if scope is None else ast.walk(scope)
        found = []
        for node in nodes:
            if isinstance(node, ast.AugAssign) and isinstance(node.target,
                                                              ast.Name) and node.target.id == name:
                return None
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                found.append(node)
        return found

    def only_called(self, expr, calls):
        """Whether every use of the value of `expr` is a call, collecting those calls."""
        if id(expr) in self.seen:
            return True
        self.seen.add(id(expr))
        parent = self.parents.get(id(expr))
        if isinstance(parent, ast.Call) and parent.func is expr:
            calls.append(parent)
            return True
        if isinstance(parent, ast.IfExp) and expr is not parent.test:
            return self.only_called(parent, calls)
        if isinstance(parent, ast.Return):
            fn = self.scope(parent)
            return bool(fn) and self.results_only_called(fn, calls)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(
                parent.targets[0], ast.Name):
            return self.name_only_called(parent.targets[0].id, self.scope(parent), calls)
        if isinstance(parent, ast.Call) and any(expr is arg for arg in parent.args):
            return self.argument_only_called(parent, expr, calls)
        return False

    def name_only_called(self, name, scope, calls):
        """Whether every read of `name`, bound in `scope`, is only called."""
        if scope is False or (scope is not None and
                              (not _is_local_to(scope, name) or name
                               in [a.arg for a in ast.walk(scope.args) if isinstance(a, ast.arg)])):
            return False
        loads = self.loads(name, scope)
        return loads is not None and all(self.only_called(load, calls) for load in loads)

    def results_only_called(self, fn, calls):
        """Whether every result of the module-level def `fn` is only called."""
        return self.calls_of_def(fn, lambda call: self.only_called(call, calls))

    def calls_of_def(self, fn, check):
        """Whether every read of the module-level def `fn` is a call that `check` accepts,
        directly, through a name or through a def returning it."""
        if fn.decorator_list or self.bindings(fn.name) != 1:
            return False
        loads = self.loads(fn.name, None)
        return loads is not None and all(self.calls_of(load, check) for load in loads)

    def calls_of(self, expr, check):
        """Whether every use of `expr`, which evaluates to a function, is a call that `check`
        accepts, directly, through a name or returned by a module-level def."""
        if ("calls", id(expr)) in self.seen:
            return True
        self.seen.add(("calls", id(expr)))
        parent = self.parents.get(id(expr))
        if isinstance(parent, ast.Call) and parent.func is expr:
            return check(parent)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(
                parent.targets[0], ast.Name):
            name, scope = parent.targets[0].id, self.scope(parent)
            if scope is False or (scope is not None and not _is_local_to(scope, name)):
                return False
            loads = self.loads(name, scope)
            return loads is not None and all(self.calls_of(load, check) for load in loads)
        if isinstance(parent, ast.Return):
            fn = self.scope(parent)
            return bool(fn) and self.calls_of_def(fn, lambda call: self.calls_of(call, check))
        return False

    def callee_def(self, func):
        """The module-level def a callee names, directly or as the call `r()` of a def that
        only returns it."""
        if isinstance(func, ast.Name):
            return self.sole_def(func.id, func)
        if not (isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and not func.args
                and not func.keywords):
            return None
        returner = self.sole_def(func.func.id, func.func)
        if returner is None or returner.args.args or len(returner.body) != 1:
            return None
        stmt = returner.body[0]
        if not (isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name)):
            return None
        return self.sole_def(stmt.value.id, stmt.value)

    def argument_only_called(self, call, arg, calls):
        """Whether the function `call` passes `arg` to only calls that parameter or returns it."""
        fn = self.callee_def(call.func)
        if fn is None or not _binds_by_position(fn, call):
            return False
        param = fn.args.args[next(i for i, a in enumerate(call.args) if a is arg)].arg
        nested = [n for n in ast.walk(fn) if n is not fn and isinstance(n, _SCOPES)]
        if any(isinstance(n, ast.Name) and n.id == param for s in nested for n in ast.walk(s)):
            return False
        return all(
            self.parameter_use_called(node, call, calls) for node in _own_nodes(fn)
            if isinstance(node, ast.Name) and node.id == param)

    def parameter_use_called(self, node, call, calls):
        """Whether `node`, a use of a parameter `call` binds, calls it or returns it from
        `call`, whose value is then only called."""
        parent = self.parents.get(id(node))
        if not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(parent, ast.Return):
            return self.only_called(call, calls)
        if isinstance(parent, ast.Call) and parent.func is node:
            calls.append(parent)
            return True
        return False


def _binds_by_position(fn, call):
    """Whether `call` binds each parameter of `fn`, which are all plain, by position."""
    args = fn.args
    return not (call.keywords or any(isinstance(a, ast.Starred)
                                     for a in call.args) or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or args.defaults) and len(args.args) == len(call.args)


def _never_callable(flow, expr):
    """Whether `expr` is a constant, or a call of a module-level def that returns only
    constants, so its value is never callable."""
    if isinstance(expr, ast.Constant):
        return True
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
        return False
    fn = flow.sole_def(expr.func.id, expr.func)
    return fn is not None and all(
        isinstance(n.value, ast.Constant) for n in _own_nodes(fn) if isinstance(n, ast.Return))


def _arity(calls):
    """The number of positional arguments every call of a call's result in `calls` passes."""
    if not calls or any(not isinstance(c.func, ast.Call) or c.keywords or any(
            isinstance(a, ast.Starred) for a in c.args) for c in calls):
        return None
    counts = {len(c.args) for c in calls}
    return counts.pop() if len(counts) == 1 else None


def _reflective(module):
    return any((isinstance(n, ast.Name) and n.id in _REFLECTIVE) or (
        isinstance(n, ast.Attribute) and n.attr in _REFLECTIVE_ATTRS) for n in ast.walk(module))


def _wrapper_defs(arity):
    params = ", ".join(f"_esbmc_p{i}" for i in range(arity))
    return ast.parse(f"def {_PREFIX}_{arity}({params}):\n"
                     f"    raise TypeError('object is not callable')\n\n\n"
                     f"def {_PREFIX}_of_{arity}(value):\n"
                     f"    return {_PREFIX}_{arity}\n").body


def _replace(parent, old, new):
    for field, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, field, new)
        elif isinstance(value, list):
            setattr(parent, field, [new if v is old else v for v in value])


def _called_values(module, flow):
    """Each never-callable value of `module` whose every use is a call of a call's result, with
    the number of arguments those calls pass."""
    found = []
    for node in ast.walk(module):
        # A constant a def returns is left to the calls of that def.
        if not _never_callable(flow, node) or isinstance(flow.parents.get(id(node)), ast.Return):
            continue
        flow.seen = set()
        calls = []
        arity = _arity(calls) if flow.only_called(node, calls) else None
        if arity is not None:
            found.append((node, arity))
    return found


def _preamble(module):
    """How many statements open `module` as its docstring and `__future__` imports."""
    count = 0
    for stmt in module.body:
        docstring = count == 0 and isinstance(stmt, ast.Expr) and isinstance(
            stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
        if not (docstring or (isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__")):
            break
        count += 1
    return count


def wrap_never_callable_values(module):
    """Replace each never-callable value whose every use is a call of a call's result."""
    if _reflective(module) or any(
            isinstance(n, ast.Name) and n.id.startswith(_PREFIX) for n in ast.walk(module)):
        return
    flow = _Flow(module)
    found = _called_values(module, flow)
    if not found:
        return
    for node, arity in found:
        wrapped = ast.Call(ast.Name(f"{_PREFIX}_of_{arity}", ast.Load()), [node], [])
        _replace(flow.parents[id(node)], node, ast.copy_location(wrapped, node))
    first = _preamble(module)
    module.body[first:first] = [
        d for arity in sorted({a
                               for _, a in found}) for d in _wrapper_defs(arity)
    ]
    ast.fix_missing_locations(module)
