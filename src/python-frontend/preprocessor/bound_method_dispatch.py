"""Call the method a name was last bound to, when it is bound at several sites (#8467).

After `if c: f = o.inc` / `else: f = p.dec`, CPython's `f()` calls the
method of whichever binding ran last. Each binding stores its receiver and its
own number in a selector, and each call becomes an if/elif over the bindings
that calls only the method the selector names. The selector starts at 0, whose
arm raises NameError, or UnboundLocalError in a function, as calling an unbound
`f` does.

A call is rewritten only as the whole value of an expression statement, an
assignment or a return, so each arm evaluates the arguments once, in order.
Any other read of the name, a binding of another form, or a binding from
another scope is refused.

A name whose bindings are all simple statements of its scope's body is left
to the per-binding receiver capture instead: there each call is reached only
by the last binding above it in the source.
"""

import ast
import copy
from collections import Counter

from preprocessor.rejection import attach_location
from preprocessor.unbound_locals import _NESTED_SCOPES, _global_error, _local_error, _located


def _scope_nodes(scope):
    """The nodes of `scope` outside its nested scopes, which are yielded but not entered."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _NESTED_SCOPES):
            stack.extend(ast.iter_child_nodes(node))


# The statements that bind a name once each time they run.
_SIMPLE_BINDINGS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom,
                    ast.FunctionDef, ast.ClassDef)


def _reads(node, name):
    return isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)


def _method_binding(node):
    """`f = obj.attr` with `obj` a name or a call, the form a call of `f` is
    rewritten for."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, (ast.Name, ast.Call)))


def _call_of(stmt, name):
    """Whether `stmt` is `name(...)`, `targets = name(...)` or `return name(...)`."""
    value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign, ast.Return)) else None
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == name)


class _Dispatch:
    """The rewrite of one name bound to methods at several sites of one scope."""

    def __init__(self, scope, name, next_id):
        self.scope = scope
        self.name = name
        self.selector = f"ESBMC_bound_sel_{next_id()}"
        self.next_id = next_id
        self.methods = []

    def bind(self, assign, variables):
        """The statements replacing `assign`: store its receiver, unless it is
        bound only by an import, a definition or a builtin, and select it."""
        self.methods.append(assign.value)
        stored = assign.value.value
        replacement = []
        if isinstance(stored, ast.Call) or stored.id in variables:
            receiver = f"ESBMC_bound_receiver_{self.next_id()}"
            replacement.append(
                ast.Assign(targets=[ast.Name(id=receiver, ctx=ast.Store())], value=stored))
            self.methods[-1] = ast.Attribute(value=ast.Name(id=receiver, ctx=ast.Load()),
                                             attr=assign.value.attr,
                                             ctx=ast.Load())
        replacement.append(self._select(len(self.methods)))
        return _located(replacement, assign)

    def call(self, stmt):
        """The if/elif replacing `stmt`, whose arm `n` calls the method of binding `n`."""
        error, message = (_global_error if isinstance(self.scope, ast.Module) else _local_error)(
            self.name)
        chain = [
            ast.Raise(exc=ast.Call(func=ast.Name(id=error, ctx=ast.Load()),
                                   args=[ast.Constant(value=message)],
                                   keywords=[]),
                      cause=None)
        ]
        for number in range(len(self.methods), 0, -1):
            arm = copy.deepcopy(stmt)
            arm.value.func = copy.deepcopy(self.methods[number - 1])
            test = ast.Compare(left=ast.Name(id=self.selector, ctx=ast.Load()),
                               ops=[ast.Eq()],
                               comparators=[ast.Constant(value=number)])
            chain = [ast.If(test=test, body=[arm], orelse=chain)]
        return _located(chain, stmt)

    def _select(self, number):
        return ast.Assign(targets=[ast.Name(id=self.selector, ctx=ast.Store())],
                          value=ast.Constant(value=number))

    def start(self):
        """Clear the selector at the start of the scope."""
        self.scope.body[:0] = _located([self._select(0)], self.scope.body[0])


def _called_names(module):
    """The names `module` calls as `name(...)`."""
    return {
        n.func.id
        for n in ast.walk(module) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def _return_types(module, methods):
    """The return annotations of the methods named `methods` in the classes of
    `module`, None standing for a method without one; empty if none is defined."""
    return {
        ast.dump(n.returns) if n.returns else None
        for cls in ast.walk(module) if isinstance(cls, ast.ClassDef) for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in methods
    }


class _Replace(ast.NodeTransformer):

    def __init__(self, replacements):
        self.replacements = replacements

    def generic_visit(self, node):
        super().generic_visit(node)
        return self.replacements.get(id(node), node)


def dispatch_rebound_methods(module, names_bound_by, variables, next_id, module_name):
    """Give each name bound to methods at several sites of a scope, and called
    somewhere, a selector set by its bindings and read by its calls. Return the
    (scope, name) pairs left alone because their bindings run in source order,
    the scope None standing for the module."""
    return _Rebound(module, names_bound_by, module_name).rewrite(variables, next_id)


class _Rebound:
    """The names of `module` bound to methods at several sites of a scope."""

    def __init__(self, module, names_bound_by, module_name):
        self.module = module
        self.names_bound_by = names_bound_by
        self.module_name = module_name

    def rewrite(self, variables, next_id):
        """Rewrite the bindings and calls of each such name."""
        called = _called_names(self.module)
        in_order = set()
        replacements = {}
        dispatches = []
        for scope in (self.module,
                      *(n for n in ast.walk(self.module) if isinstance(n, ast.FunctionDef))):
            nodes = list(_scope_nodes(scope))
            counts = Counter(name for n in nodes for name in self._binds(n))
            for name in sorted({n.targets[0].id for n in nodes if _method_binding(n)} & called):
                if counts[name] < 2:
                    continue
                if self._in_order(scope, name, nodes):
                    in_order.add((None if scope is self.module else id(scope), name))
                    continue
                assigns, calls = self.dispatched_statements(scope, name, nodes)
                dispatch = _Dispatch(scope, name, next_id)
                for assign in assigns:
                    replacements[id(assign)] = dispatch.bind(assign, variables)
                replacements.update((id(call), dispatch.call(call)) for call in calls)
                dispatches.append(dispatch)
        _Replace(replacements).visit(self.module)
        for dispatch in dispatches:
            dispatch.start()
        return in_order

    def _in_order(self, scope, name, nodes):
        """Whether each binding of `name` is a simple statement of the body of
        `scope`, and only `scope` reads it, so a call is reached only by the
        last binding above it in the source."""
        top = {
            id(n)
            for stmt in scope.body if isinstance(stmt, _SIMPLE_BINDINGS) for n in ast.walk(stmt)
        }
        own = {id(n) for n in nodes}
        return all(id(n) in top for n in nodes if name in self._binds(n)) and not any(
            self._other_use(n, name, own, scope) for n in ast.walk(self.module))

    def _binds(self, node):
        """The names `node` binds; `global` and `nonlocal` only say where."""
        return [] if isinstance(node, (ast.Global, ast.Nonlocal)) else self.names_bound_by(node)

    def dispatched_statements(self, scope, name, nodes):
        """The bindings and calls of `name` in `scope`, after refusing what the
        rewrite cannot make exact."""
        assigns = {id(n.targets[0]): n for n in nodes if _method_binding(n)}
        bindings = sorted((n for n in nodes if name in self._binds(n)),
                          key=lambda n: (n.lineno, n.col_offset))
        for binding in bindings:
            if id(binding) not in assigns:
                self._refuse(binding, f"'{name}' is bound to a method and also bound another way")
        calls = [n for n in nodes if _call_of(n, name)]
        return_types = _return_types(self.module, {assigns[id(b)].value.attr for b in bindings})
        for call in calls:
            self._check_call(call, name, bindings[-1], return_types)
        reached = {id(call.value.func) for call in calls}
        for n in ast.walk(self.module):
            if self._other_use(n, name, reached, scope):
                self._refuse(
                    n, f"'{name}' is bound to methods at several sites and used other than as "
                    "a whole call statement")
        return [assigns[id(b)] for b in bindings], calls

    def _other_use(self, node, name, expected, scope):
        """Whether `node` reads `name` and is not among the `expected` reads, or
        makes another scope bind it."""
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return name in node.names
        return (_reads(node, name) and id(node) not in expected
                and not self._bound_around(node, scope, name))

    def _check_call(self, call, name, last_binding, return_types):
        """Refuse a call above a binding, whose receiver has no type there yet,
        or one whose result is kept while the methods may differ in type."""
        if (call.lineno, call.col_offset) < (last_binding.lineno, last_binding.col_offset):
            self._refuse(call, f"'{name}' is called above one of its method bindings")
        if not isinstance(call, ast.Expr) and (len(return_types) != 1 or None in return_types):
            self._refuse(call, f"the methods '{name}' is bound to may return different types")

    def _bound_around(self, read, scope, name):
        """Whether `read` is directly in a scope other than `scope` that binds `name`."""
        return any(other is not scope and any(n is read for n in _scope_nodes(other)) and any(
            name in self._binds(n) for n in _scope_nodes(other)) for other in ast.walk(self.module)
                   if isinstance(other, (ast.Module, *_NESTED_SCOPES)))

    def _refuse(self, node, message):
        exc = NotImplementedError(message)
        attach_location(exc, node, self.module_name)
        raise exc
