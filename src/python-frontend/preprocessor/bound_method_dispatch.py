"""Call the method a name was last bound to, when it is bound at several sites (#8467).

After `if c: f = o.inc` / `else: f = p.dec`, CPython's `f()` calls the
method of whichever binding ran last. Each binding stores its receiver and its
own number in a selector, and each call becomes an if/elif over the bindings
that calls only the method the selector names. The selector starts at 0, whose
arm raises NameError, or UnboundLocalError in a function, as calling an unbound
`f` does.

A call is rewritten only as the whole value of an expression statement, an
assignment or a return, or inside an assert, so each arm evaluates the
statement once, in order.
Any other read of the name fails an assertion just before the statement that
holds it, so a read in code that never runs does not stop verification (#9160).
A plain `def` of the name is one more binding, renamed and called like a
method. A decorated or async `def`, or a class, fails an assertion where it
would run. A binding of another form, or a binding from another scope, is
refused.

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


def _plain_function(node):
    return isinstance(node, ast.FunctionDef) and not node.decorator_list


def _bound_value(binding):
    if isinstance(binding, ast.FunctionDef):
        return ast.Name(id=binding.name, ctx=ast.Load())
    return binding.value


def _is_call_of(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _calls_of(stmt, name):
    """The calls of `name` when `stmt` is `name(...)`, `targets = name(...)`,
    `return name(...)` or an assert holding them, else []."""
    if isinstance(stmt, ast.Assert):
        return [n for n in _scope_nodes(stmt) if _is_call_of(n, name)]
    value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign, ast.Return)) else None
    return [value] if _is_call_of(value, name) else []


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
        bound only by an import, a definition or a builtin, and select it. A
        `def` is renamed and selected after it."""
        if isinstance(assign, ast.FunctionDef):
            assign.name = f"ESBMC_bound_def_{self.next_id()}"
            self.methods.append(ast.Name(id=assign.name, ctx=ast.Load()))
            return [assign] + _located([self._select(len(self.methods))], assign)
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
            for call in _calls_of(arm, self.name):
                call.func = copy.deepcopy(self.methods[number - 1])
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


def _return_types(module, bindings):
    """The return annotations of the `def` bindings and of the methods the
    other bindings name in the classes of `module`, None standing for one
    without an annotation; empty if no method is defined."""
    methods = {b.value.attr for b in bindings if isinstance(b, ast.Assign)}
    functions = [b for b in bindings if isinstance(b, ast.FunctionDef)]
    return {
        ast.dump(n.returns) if n.returns else None
        for n in functions + [
            n for cls in ast.walk(module) if isinstance(cls, ast.ClassDef)
            for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
    }


def _own_expressions(stmt):
    """The expressions of `stmt` outside the statements it holds."""
    stack = [c for c in ast.iter_child_nodes(stmt) if not isinstance(c, ast.stmt)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, ast.stmt))


def _failed_assert(message):
    args = [ast.Constant(value=False), ast.Constant(value=message)]
    return ast.Expr(
        value=ast.Call(func=ast.Name(id="__ESBMC_assert", ctx=ast.Load()), args=args, keywords=[]))


def _refuse_when_run(module, reads, method, message, replacements):
    """Make each statement holding one of `reads` fail an assertion first, so
    only a read that may run makes the program fail, and read `method` there
    so the statement still has a type."""
    reads = {id(n) for n in reads}
    for stmt in (n for n in ast.walk(module) if isinstance(n, ast.stmt)):
        own = [n for n in _own_expressions(stmt) if id(n) in reads]
        if not own:
            continue
        replacements.update((id(n), copy.deepcopy(method)) for n in own)
        replacements[id(stmt)] = _located([_failed_assert(message)], stmt) + replacements.get(
            id(stmt), [stmt])


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
                assigns, calls = self.dispatched_statements(name, nodes, replacements)
                dispatch = _Dispatch(scope, name, next_id)
                for assign in assigns:
                    replacements[id(assign)] = dispatch.bind(assign, variables)
                replacements.update((id(call), dispatch.call(call)) for call in calls)
                _refuse_when_run(
                    self.module, self._other_reads(scope, name, calls), _bound_value(assigns[-1]),
                    f"'{name}' is bound to methods at several sites and used other than as "
                    "a whole call statement", replacements)
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

    def dispatched_statements(self, name, nodes, replacements):
        """The bindings and calls of `name` among `nodes`, after refusing what the
        rewrite cannot make exact, and making a class or another `def` of it
        fail an assertion where it would run."""
        assigns = {id(n.targets[0]): n for n in nodes if _method_binding(n)}
        assigns.update((id(n), n) for n in nodes if _plain_function(n))
        bindings = sorted((n for n in nodes if name in self._binds(n)),
                          key=lambda n: (n.lineno, n.col_offset))
        message = f"'{name}' is bound to a method and also bound another way"
        for binding in bindings:
            if id(binding) in assigns:
                continue
            if not isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._refuse(binding, message)
            replacements[id(binding)] = _located([_failed_assert(message)], binding)
        bindings = [assigns[id(b)] for b in bindings if id(b) in assigns]
        calls = [n for n in nodes if _calls_of(n, name)]
        return_types = _return_types(self.module, bindings)
        for call in calls:
            self._check_call(call, name, bindings[-1], return_types)
        return bindings, calls

    def _other_reads(self, scope, name, calls):
        """The reads of `name` other than by `calls`, after refusing a binding
        of it from another scope."""
        expected = {id(c.func) for call in calls for c in _calls_of(call, name)}
        reads = []
        for n in ast.walk(self.module):
            if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
                self._refuse(
                    n, f"'{name}' is bound to methods at several sites and used other than as "
                    "a whole call statement")
            if self._other_read(n, name, expected, scope):
                reads.append(n)
        return reads

    def _other_use(self, node, name, expected, scope):
        """Whether `node` reads `name` and is not among the `expected` reads, or
        makes another scope bind it."""
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return name in node.names
        return self._other_read(node, name, expected, scope)

    def _other_read(self, node, name, expected, scope):
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
