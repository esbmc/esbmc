"""Refuse a call through a function value that may return different types (#8404).

A call such as `g()`, where `g` may hold one of several functions, takes one
return type in the converter. When those functions return values of different
types, the result has the wrong type on every path but one, so `isinstance`
and comparisons give verdicts CPython disagrees with. Until the result can
carry its type per path, such a call is rejected.

The functions a value may hold are found by a flow-insensitive analysis over
one module: names, parameters, returns and container elements that a function
value flows through are merged into one group. A name is keyed by the scope
that binds it. A call through a value merges each function of its group's
arguments into that function's parameters and its returns into the call's
result, until nothing changes. A function whose return type cannot be told
from its source never causes a rejection.
"""

import ast
import os

_NUMERIC = {"bool", "int", "float", "num"}
_NONDET_PREFIX = "nondet_"
_SHOWN = {"num": "int or float", "function": "a function"}
# Operators no str, list or set supports, so their result is a number.
_NUMERIC_ONLY_OPS = (ast.FloorDiv, ast.Pow, ast.LShift, ast.RShift)
_BUILTIN_RESULTS = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "len": "int",
    "repr": "str",
    "list": "list",
    "dict": "dict",
    "tuple": "tuple",
    "set": "set"
}


def _differ(a, b):
    """Whether two known return kinds name different types."""
    if a == b:
        return False
    return not ("num" in (a, b) and {a, b} <= _NUMERIC)


class _Scope:
    """A function, lambda or the module, with the names it binds."""

    def __init__(self, node, parent):
        self.node = node
        self.parent = parent
        self.names = set()
        args = getattr(node, "args", None)
        if args is not None:
            self.params = [a.arg for a in args.posonlyargs + args.args]
            self.keywords = self.params[len(args.posonlyargs):] + [a.arg for a in args.kwonlyargs]
            self.names.update(self.params, self.keywords)
        else:
            self.params = []
            self.keywords = []
        declared = set()
        for sub in _own_nodes(node):
            if isinstance(sub, (ast.Global, ast.Nonlocal)):
                declared.update(sub.names)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                self.names.add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.names.add(sub.name)
        self.names -= declared


def _own_nodes(node):
    """The nodes of `node`'s own scope, with nested definitions but not their bodies."""
    body = node.body if isinstance(node.body, list) else [node.body]
    stack = list(body)
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Defaults are evaluated in the enclosing scope.
            stack.extend(current.args.defaults)
            stack.extend(d for d in current.args.kw_defaults if d is not None)
            continue
        if isinstance(current, ast.ClassDef):
            continue
        stack.extend(ast.iter_child_nodes(current))


class _Analysis:
    """The groups of values that function values flow through in one module."""

    def __init__(self, module, path):
        self.dir = os.path.dirname(os.path.abspath(path)) if path else ""
        self.parent = {}
        self.members = {}
        self.calls = []
        self.scopes = {}
        self.defs = {}
        self.kinds = {}
        self.imports = {}
        self.kind_stack = set()
        self.seen_calls = set()
        self.from_imports = set()
        self.loaded = {}
        self.rebound = {}
        self._collect_scopes(module, None)
        for scope in list(self.scopes.values()):
            for node in _own_nodes(scope.node):
                self._statement(node, scope)
            if isinstance(scope.node, ast.Lambda):
                self.union(self.result_key(scope.node), self.value(scope.node.body, scope))

    # union-find
    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        if a is None or b is None:
            return False
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.members.setdefault(rb, {}).update(self.members.pop(ra, {}))
        return True

    def add_member(self, key, node, name, kind):
        self.members.setdefault(self.find(key), {})[id(node)] = (node, name, kind)

    # scopes and keys
    def _collect_scopes(self, node, parent):
        scope = _Scope(node, parent)
        self.scopes[id(node)] = scope
        for sub in _own_nodes(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(self._binding_scope(sub.name, scope), []).append(sub)
                self._collect_scopes(sub, scope)
            elif isinstance(sub, ast.Lambda):
                self._collect_scopes(sub, scope)
            elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and self._local_module(
                    sub.module or ""):
                self.from_imports.update(alias.asname or alias.name for alias in sub.names)
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    root = alias.name.split(".")[0]
                    self.imports[alias.asname or root] = alias.name if alias.asname else root
        return scope

    def _local_module(self, dotted):
        """The source file of a module next to the program, or None."""
        base = os.path.join(self.dir, *dotted.split("."))
        for path in (base + ".py", os.path.join(base, "__init__.py")):
            if self.dir and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _binding_scope(name, scope):
        while scope.parent is not None and name not in scope.names:
            scope = scope.parent
        return (id(scope.node), name)

    def name_key(self, name, scope):
        return ("name", ) + self._binding_scope(name, scope)

    @staticmethod
    def result_key(fn):
        return ("result", id(fn))

    def param_key(self, fn, index):
        scope = self.scopes[id(fn)]
        if index >= len(scope.params):
            return None
        return ("name", id(fn), scope.params[index])

    def keyword_key(self, fn, name):
        """The key of `fn`'s parameter that a keyword argument `name` binds, or None."""
        if name not in self.scopes[id(fn)].keywords:
            return None
        return ("name", id(fn), name)

    # values
    def value(self, expr, scope):
        """The key of the group `expr`'s value belongs to, or None."""
        if isinstance(expr, ast.Name):
            key = self.name_key(expr.id, scope)
            for fn in self.defs.get(key[1:], []):
                self.add_member(key, fn, fn.name, self.def_kind(fn))
            return key
        if isinstance(expr, ast.Lambda):
            key = ("lambda", id(expr))
            self.add_member(key, expr, f"the lambda on line {expr.lineno}",
                            self.expr_kind(expr.body, self.scopes[id(expr)]))
            return key
        if isinstance(expr, ast.IfExp):
            body = self.value(expr.body, scope)
            orelse = self.value(expr.orelse, scope)
            self.union(body, orelse)
            return body or orelse
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            key = ("elements", id(expr))
            for elt in expr.elts:
                self.union(key, self.value(elt, scope))
            return key
        if isinstance(expr, ast.Subscript):
            return self.value(expr.value, scope)
        if isinstance(expr, ast.Attribute):
            return self.imported_value(expr)
        if isinstance(expr, ast.Call):
            return self.call(expr, scope)
        return None

    def imported_value(self, expr):
        """The key of `module.function`, for a module imported by its path."""
        dotted = []
        node = expr
        while isinstance(node, ast.Attribute):
            dotted.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.imports or not self.dir:
            return None
        path = self._local_module(".".join(self.imports[node.id].split(".") +
                                           list(reversed(dotted[1:]))))
        if path is None:
            return None
        if path not in self.loaded:
            with open(path, encoding="utf-8") as src:
                self.loaded[path] = ast.parse(src.read())
        for stmt in self.loaded[path].body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == expr.attr:
                key = ("imported", path, stmt.name)
                self.add_member(key, stmt, f"{self.imports[node.id]}.{stmt.name}",
                                self.def_kind(stmt))
                return key
        return None

    def call(self, expr, scope):
        """Record a call, linking a direct call's arguments to the callee's parameters."""
        self.seen_calls.add(id(expr))
        if isinstance(expr.func, ast.Name):
            callees = self.defs.get(self.name_key(expr.func.id, scope)[1:], [])
            if callees and not self._rebound(expr.func.id, scope):
                for fn in callees:
                    self.link_call(fn, expr, scope)
                    self.union(self.result_key(callees[0]), self.result_key(fn))
                return self.result_key(callees[0])
        callee = self.value(expr.func, scope)
        args = [self.value(arg, scope) for arg in expr.args]
        keywords = {kw.arg: self.value(kw.value, scope) for kw in expr.keywords if kw.arg}
        if isinstance(expr.func, ast.Name) and expr.func.id in self.from_imports:
            # The imported function may call what it is given, out of sight.
            self.calls.extend((expr, arg, [], {}, None) for arg in args + list(keywords.values())
                              if arg is not None)
            return None
        if callee is None:
            return None
        result = ("call", id(expr))
        self.calls.append((expr, callee, args, keywords, result))
        return result

    def _rebound(self, name, scope):
        """Whether `name` is bound other than by its def."""
        key = self._binding_scope(name, scope)
        if key not in self.rebound:
            self.rebound[key] = any(
                isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
                for node in _own_nodes(self.scopes[key[0]].node))
        return self.rebound[key]

    def link_call(self, fn, call, scope):
        args = [self.value(arg, scope) for arg in call.args]
        keywords = {kw.arg: self.value(kw.value, scope) for kw in call.keywords if kw.arg}
        self.link_args(fn, args, keywords)

    def link_args(self, fn, args, keywords):
        """Merge argument groups into `fn`'s parameters; whether any group changed."""
        changed = False
        for i, arg in enumerate(args):
            changed |= self.union(self.param_key(fn, i), arg)
        for name, arg in keywords.items():
            changed |= self.union(self.keyword_key(fn, name), arg)
        return changed

    def _statement(self, node, scope):
        if isinstance(node, ast.Assign):
            value = self.value(node.value, scope)
            for target in node.targets:
                self._bind(target, value, node.value, scope)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self._bind(node.target, self.value(node.value, scope), node.value, scope)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self._bind(node.target, self.value(node.iter, scope), None, scope)
        elif isinstance(node, ast.Return) and node.value is not None:
            self.union(self.result_key(scope.node), self.value(node.value, scope))
        elif isinstance(node, ast.Call) and id(node) not in self.seen_calls:
            self.value(node, scope)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            self._link_defaults(node, scope)

    def _link_defaults(self, fn, scope):
        """Merge each default, evaluated in the enclosing scope, into its parameter."""
        first = len(self.scopes[id(fn)].params) - len(fn.args.defaults)
        for i, default in enumerate(fn.args.defaults):
            self.union(self.param_key(fn, first + i), self.value(default, scope))
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if default is not None:
                self.union(self.keyword_key(fn, arg.arg), self.value(default, scope))

    def _bind(self, target, value, value_node, scope):
        if isinstance(target, ast.Name):
            self.union(self.name_key(target.id, scope), value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value_node,
                          (ast.Tuple, ast.List)) and len(value_node.elts) == len(target.elts):
                for sub, elt in zip(target.elts, value_node.elts):
                    self._bind(sub, self.value(elt, scope), elt, scope)
            else:
                for sub in target.elts:
                    self._bind(sub, value, None, scope)

    # return kinds
    def def_kind(self, fn):
        """The type `fn` returns, or None when the source does not tell."""
        if id(fn) in self.kinds:
            return self.kinds[id(fn)]
        if fn.returns is not None:
            kind = ast.unparse(fn.returns)
        elif id(fn) in self.kind_stack:
            return None
        else:
            self.kind_stack.add(id(fn))
            scope = self.scopes.get(id(fn)) or self._collect_scopes(fn, None)
            returns = [n for n in _own_nodes(fn) if isinstance(n, ast.Return)]
            kinds = {
                self.expr_kind(r.value, scope) if r.value is not None else "None"
                for r in returns
            }
            if not returns:
                kinds = {"None"}
            self.kind_stack.discard(id(fn))
            kind = kinds.pop() if len(kinds) == 1 else None
        self.kinds[id(fn)] = kind
        return kind

    def expr_kind(self, expr, scope):
        """The type `expr` evaluates to, or None when the source does not tell."""
        if isinstance(expr, ast.Constant):
            return "None" if expr.value is None else type(expr.value).__name__
        if isinstance(expr, ast.JoinedStr):
            return "str"
        if isinstance(expr, (ast.Lambda, ast.FunctionDef)):
            return "function"
        if isinstance(expr, ast.BinOp):
            return self._binop_kind(expr, scope)
        if isinstance(expr, ast.Name):
            return self._name_kind(expr.id, scope)
        if isinstance(expr, ast.Call):
            return self._call_kind(expr, scope)
        return None

    def _binop_kind(self, expr, scope):
        left = self.expr_kind(expr.left, scope)
        right = self.expr_kind(expr.right, scope)
        if "str" in (left, right):
            return "str"
        known = {k for k in (left, right) if k in _NUMERIC}
        if isinstance(expr.op, ast.Div):
            return "float"
        if isinstance(expr.op, _NUMERIC_ONLY_OPS) and not known:
            return "num"
        if None in (left, right):
            return "num" if known else None
        if {left, right} <= _NUMERIC:
            return "float" if "float" in (left, right) else "int"
        return None

    def _name_kind(self, name, scope):
        key = self._binding_scope(name, scope)
        fns = self.defs.get(key, [])
        if fns:
            return "function"
        owner = self.scopes[key[0]]
        if name in owner.params:
            return None
        values = [
            n.value for n in _own_nodes(owner.node) if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ]
        if len(values) != 1 or self._has_other_bindings(owner, name):
            return None
        return self.expr_kind(values[0], owner)

    @staticmethod
    def _has_other_bindings(owner, name):
        for node in _own_nodes(owner.node):
            if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)) and isinstance(
                    node.target, ast.Name) and node.target.id == name:
                return True
        return False

    def _call_kind(self, expr, scope):
        func = expr.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name is None:
            return None
        if name.startswith(_NONDET_PREFIX):
            return name[len(_NONDET_PREFIX):]
        if isinstance(func, ast.Name):
            fns = self.defs.get(self._binding_scope(name, scope), [])
            if len(fns) == 1:
                return self.def_kind(fns[0])
            if not fns and name in _BUILTIN_RESULTS:
                return _BUILTIN_RESULTS[name]
        return None

    # fixpoint and check
    def propagate(self):
        changed = True
        while changed:
            changed = False
            for _, callee, args, keywords, result in self.calls:
                if result is None:
                    continue
                for node, _, _ in list(self.members.get(self.find(callee), {}).values()):
                    changed |= self.union(result, self.result_key(node))
                    if id(node) in self.scopes:
                        changed |= self.link_args(node, args, keywords)

    def mixed_call(self):
        """A call through a value whose functions return different types, with two of them."""
        for call, callee, _, _, result in self.calls:
            seen = []
            members = self.members.get(self.find(callee), {}).values()
            for _, name, kind in sorted(members, key=lambda m: m[0].lineno):
                if kind is None:
                    continue
                for other_name, other_kind in seen:
                    if _differ(kind, other_kind):
                        return call, result is None, (other_name, other_kind), (name, kind)
                seen.append((name, kind))
        return None


def reject_mixed_function_values(module, path):
    """Raise NotImplementedError at a call through a value whose functions disagree."""
    analysis = _Analysis(module, path)
    analysis.propagate()
    found = analysis.mixed_call()
    if found is None:
        return
    call, passed, (first, first_kind), (second, second_kind) = found
    what = "a function passed here" if passed else "this call"
    first_kind, second_kind = (_SHOWN.get(k, k) for k in (first_kind, second_kind))
    error = NotImplementedError(
        f"unsupported: {what} may be '{first}' or '{second}', which return different types "
        f"({first_kind} and {second_kind})")
    error.esbmc_location = (path, call.lineno, call.col_offset)
    raise error
