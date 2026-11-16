"""Give a NumPy array name rebound on some paths one variable per binding.

The converter's symbol for a name holds one array type, so a rebind to another
shape or dtype on a path, or one a function reads, cannot retype it. Each binding gets its own variable
instead, a tag records the one that ran last, and a statement reading the name
where the tag is not known is repeated for each binding:

    a = np.array([1, 2])         esbmc_np_a_0 = np.array([1, 2])
                                 esbmc_np_a_1 = np.array([5, 6, 7])
                                 esbmc_np_a_tag = 0
    if c:                        if c:
        a = np.array([5, 6, 7])      esbmc_np_a_1 = np.array([5, 6, 7])
                                     esbmc_np_a_tag = 1
    n = len(a)                   if esbmc_np_a_tag == 0:
                                     n = len(esbmc_np_a_0)
                                 else:
                                     n = len(esbmc_np_a_1)

A view of an earlier binding keeps pointing at that binding's variable, which a
later binding does not write. Only constructors with literal arguments qualify, as their variables must exist
on every path. Any other form leaves the scope unchanged, for the converter to
accept or reject.
"""

import ast
import copy

_CONSTRUCTORS = {"array", "zeros", "ones", "full", "arange", "eye", "identity", "linspace"}
_DTYPES = {"int", "float", "bool", "complex"}
_SCALAR_ATTRIBUTES = {"ndim", "size"}
_SCALAR_BUILTINS = {"len", "int", "float", "bool"}
_REDUCERS = {"sum", "mean", "min", "max", "any", "all", "item"}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = _FUNCTIONS + (ast.Lambda, ast.ClassDef)
_COMPOUND = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith, ast.Match)


class _Unsupported(Exception):
    """The scope uses the name in a way the rewrite cannot keep."""


def _numpy_aliases(module):
    return {
        alias.asname or alias.name
        for stmt in module.body if isinstance(stmt, ast.Import) for alias in stmt.names
        if alias.name == "numpy"
    }


def _is_literal(node, aliases):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_literal(e, aliases) for e in node.elts)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.USub, ast.UAdd)) and _is_literal(node.operand, aliases)
    if isinstance(node, ast.Name):
        return node.id in _DTYPES
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in aliases)


def _is_constructor(node, aliases):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases
            and node.func.attr in _CONSTRUCTORS and all(_is_literal(a, aliases) for a in node.args)
            and all(k.arg and _is_literal(k.value, aliases) for k in node.keywords))


def _layout(call):
    """The constructor without its element values, which do not change the layout."""
    shape = copy.deepcopy(call)
    if call.func.attr == "array":
        for node in ast.walk(shape.args[0] if shape.args else shape):
            if isinstance(node, ast.Constant):
                node.value = type(node.value).__name__
    return ast.dump(shape)


def _rank(call):
    """The rank of the array `call` makes."""
    if call.func.attr in ("eye", "identity"):
        return 2
    first = call.args[0] if call.args else None
    if call.func.attr == "array":
        rank = 0
        while isinstance(first, (ast.List, ast.Tuple)) and first.elts:
            rank += 1
            first = first.elts[0]
        return rank
    if call.func.attr in ("arange", "linspace") or not isinstance(first, ast.Tuple):
        return 1
    return len(first.elts)


def _operands(node):
    """The operands of `node` if it is scalar when they all are, or None."""
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if isinstance(node, ast.Compare):
        return [node.left] + node.comparators
    if isinstance(node, ast.IfExp):
        return [node.test, node.body, node.orelse]
    return None


def _is_scalar_call(node):
    if not isinstance(node, ast.Call) or node.keywords:
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in _SCALAR_BUILTINS
    return (isinstance(node.func, ast.Attribute) and not node.args
            and isinstance(node.func.value, ast.Name) and node.func.attr in _REDUCERS)


def _is_scalar(node, name, rank):
    """Whether `node` is a scalar on every binding of `name`, of rank `rank` at most."""
    operands = _operands(node)
    if operands is not None:
        return all(_is_scalar(n, name, rank) for n in operands)
    if isinstance(node, ast.Name):
        return node.id != name
    if isinstance(node, ast.Subscript):
        return (isinstance(node.value, ast.Name) and node.value.id == name and rank == 1
                and not isinstance(node.slice, (ast.Slice, ast.Tuple, ast.List)))
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.attr in _SCALAR_ATTRIBUTES
    return isinstance(node, ast.Constant) or _is_scalar_call(node)


def _walk_scope(stmts):
    """Every node of `stmts`, without entering nested scopes."""
    stack = list(stmts)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _is_site(stmt, name):
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name)


def _reads(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _deferred_read(stmt, name):
    """Whether `stmt` holds code reading `name` after the statement has run."""
    return any(
        isinstance(n, (ast.Lambda, ast.GeneratorExp)) and _reads(n, name) for n in ast.walk(stmt))


def _binds_otherwise(node, name, sites):
    """Whether `node` binds `name` other than by one of the `sites`."""
    if isinstance(node, ast.Name):
        return node.id == name and not isinstance(node.ctx, ast.Load) and not any(
            s.targets[0] is node for s in sites)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.alias):
        return (node.asname or node.name) == name
    return isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names


def _is_param(function, name):
    args = function.args
    params = args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
    return any(p and p.arg == name for p in params)


def _has_own_binding(function, name):
    """Whether `function` has its own `name`, so that it does not read the module's."""
    return _is_param(function, name) or any(
        isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)
        for n in _walk_scope(function.body))


class _Rewriter:
    """Rewrites the reads and bindings of one name in one scope."""

    def __init__(self, name, sites, reads_in_functions):
        self.name = name
        self.reads_in_functions = reads_in_functions
        self.sites = sites
        self.rank = max(_rank(site.value) for site in sites)
        self.tag = f"esbmc_np_{name}_tag"
        self.slots = [f"esbmc_np_{name}_{i}" for i in range(len(sites))]
        self.ids = {id(site): i for i, site in enumerate(sites)}
        self.tests = 0
        self.rebinding_loops = 0

    def _store(self, target, value, at):
        assign = ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=value)
        return ast.fix_missing_locations(ast.copy_location(assign, at))

    def preamble(self, at):
        """Bind every variable before the first statement that may need it."""
        stmts = [self._store(s, copy.deepcopy(c.value), at) for s, c in zip(self.slots, self.sites)]
        self.may_be_unbound = at is not self.sites[0]
        stmts.append(self._store(self.tag, ast.Constant(-1 if self.may_be_unbound else 0), at))
        return stmts

    def _rename(self, stmt, index):
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == self.name:
                node.id = self.slots[index]
        return stmt

    def _unbound_error(self, at):
        message = f"cannot access local variable '{self.name}'"
        error = ast.Call(func=ast.Name(id="UnboundLocalError", ctx=ast.Load()),
                         args=[ast.Constant(message)],
                         keywords=[])
        return ast.fix_missing_locations(ast.copy_location(ast.Raise(exc=error), at))

    def _reject_array_binding(self, stmt):
        """What `stmt` binds or returns would hold another array on each path."""
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Return)) else None
        if value is not None and not _is_scalar(value, self.name, self.rank):
            raise _Unsupported()

    def _reject_escape_from_loop(self, stmt):
        """A binding in a loop reuses its variable, which an escaped array would see."""
        if self.rebinding_loops > 0:
            self._reject_array_binding(stmt)
            if isinstance(stmt, ast.Expr):
                raise _Unsupported()

    def _dispatch(self, stmt):
        """`stmt` once per binding, under a test of the tag."""
        self._reject_array_binding(stmt)
        chain = [self._unbound_error(stmt)] if self.may_be_unbound else []
        last = len(self.slots) - 1
        for index in range(last, -1, -1):
            body = [self._rename(copy.deepcopy(stmt), index)]
            if not chain and index == last:
                chain = body
                continue
            test = ast.Compare(left=ast.Name(id=self.tag, ctx=ast.Load()),
                               ops=[ast.Eq()],
                               comparators=[ast.Constant(index)])
            chain = [ast.If(test=test, body=body, orelse=chain)]
        ast.fix_missing_locations(ast.copy_location(chain[0], stmt))
        return chain

    def _hoist_test(self, stmt):
        """Read the test of `stmt` into a variable set per binding."""
        self.tests += 1
        variable = f"esbmc_np_{self.name}_test_{self.tests}"
        hoisted = self._dispatch(self._store(variable, stmt.test, stmt))
        stmt.test = ast.fix_missing_locations(
            ast.copy_location(ast.Name(id=variable, ctx=ast.Load()), stmt.test))
        return hoisted

    def _lower_if(self, stmt, known):
        before = []
        if _reads(stmt.test, self.name):
            if known is None:
                before = self._hoist_test(stmt)
            else:
                self._rename(stmt.test, known)
        stmt.body, after_body = self.lower(stmt.body, known)
        stmt.orelse, after_else = self.lower(stmt.orelse, known)
        return before + [stmt], after_body if after_body == after_else else None

    def _lower_loop(self, stmt, known):
        has_site = any(id(n) in self.ids for n in ast.walk(stmt))
        header = stmt.iter if isinstance(stmt, (ast.For, ast.AsyncFor)) else stmt.test
        if _reads(header, self.name):
            # A while test runs again after the body, where the tag may differ.
            if known is None or (has_site and isinstance(stmt, ast.While)):
                raise _Unsupported()
            self._rename(header, known)
        entry = None if has_site else known
        self.rebinding_loops += has_site
        stmt.body, _ = self.lower(stmt.body, entry)
        stmt.orelse, _ = self.lower(stmt.orelse, entry)
        self.rebinding_loops -= has_site
        return [stmt], entry

    def _lower_compound(self, stmt, known):
        if isinstance(stmt, ast.If):
            return self._lower_if(stmt, known)
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            return self._lower_loop(stmt, known)
        if not _reads(stmt, self.name):
            return [stmt], known
        raise _Unsupported()

    def lower_scope(self, stmt):
        """A function reads whichever binding the tag holds when it is called."""
        if not _reads(stmt, self.name):
            return stmt
        if not self.reads_in_functions or not isinstance(stmt, _FUNCTIONS):
            raise _Unsupported()
        # A default or a decorator runs where the function is defined, outside its body.
        if any(_reads(n, self.name) for n in ast.iter_child_nodes(stmt) if n not in stmt.body):
            raise _Unsupported()
        if not _has_own_binding(stmt, self.name):
            stmt.body, _ = self.lower(stmt.body, None)
        return stmt

    def lower(self, stmts, known):
        """Rewrite `stmts`, entered with binding `known` (None when unknown)."""
        lowered = []
        for stmt in stmts:
            if id(stmt) in self.ids:
                known = self.ids[id(stmt)]
                lowered += [
                    self._store(self.slots[known], stmt.value, stmt),
                    self._store(self.tag, ast.Constant(known), stmt)
                ]
            elif isinstance(stmt, _COMPOUND):
                new, known = self._lower_compound(stmt, known)
                lowered += new
            elif isinstance(stmt, _SCOPES):
                lowered.append(self.lower_scope(stmt))
            elif not _reads(stmt, self.name):
                lowered.append(stmt)
            elif _deferred_read(stmt, self.name):
                raise _Unsupported()
            elif known is not None:
                self._reject_escape_from_loop(stmt)
                lowered.append(self._rename(stmt, known))
            else:
                self._reject_escape_from_loop(stmt)
                lowered += self._dispatch(stmt)
        return lowered, known


def _sites(scope_body, name, aliases):
    """The bindings of `name` in the scope, or None unless all are constructors."""
    sites = [s for s in _walk_scope(scope_body) if _is_site(s, name)]
    if not all(_is_constructor(s.value, aliases) for s in sites):
        return None
    if any(_binds_otherwise(n, name, sites) for n in _walk_scope(scope_body)):
        return None
    if any(
            isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names for stmt in scope_body
            for n in ast.walk(stmt)):
        return None
    sites.sort(key=lambda s: (s.lineno, s.col_offset))
    return sites


def _keeps_storage_elsewhere(scope_body, name):
    """Whether a statement of the scope may leave an array over the storage of `name`."""
    return any(
        isinstance(n, ast.Assign) and _reads(n.value, name) and not _is_scalar(n.value, name, 1)
        for n in _walk_scope(scope_body))


def _read_in_function(scope_body, name):
    return any(
        isinstance(n, _FUNCTIONS) and _reads(n, name) and not _has_own_binding(n, name)
        for stmt in scope_body for n in ast.walk(stmt))


def _needs_rewrite(scope, sites, name):
    differs = len({_layout(s.value) for s in sites}) > 1
    if isinstance(scope, ast.Module) and differs and _read_in_function(scope.body, name):
        return True
    if all(any(s is top for top in scope.body) for s in sites):
        return False
    return differs or _keeps_storage_elsewhere(scope.body, name)


def _lower_name(scope, name, aliases):
    # A parameter is a binding the rewrite has no constructor for.
    if isinstance(scope, _FUNCTIONS) and _is_param(scope, name):
        return
    sites = _sites(scope.body, name, aliases)
    if not sites or not _needs_rewrite(scope, sites, name):
        return
    first = next(i for i, stmt in enumerate(scope.body)
                 if any(n is sites[0] for n in ast.walk(stmt)))
    rewriter = _Rewriter(name, sites, isinstance(scope, ast.Module))
    preamble = rewriter.preamble(scope.body[first])
    # A failed rewrite must leave the scope as it was: only the copy is touched.
    original = copy.deepcopy(scope.body)
    try:
        before = [
            rewriter.lower_scope(s) if isinstance(s, _SCOPES) else s for s in scope.body[:first]
        ]
        body, _ = rewriter.lower(scope.body[first:], None)
    except _Unsupported:
        scope.body = original
        return
    scope.body = before + preamble + body


def _candidate_names(scope_body):
    names = {
        s.targets[0].id
        for s in _walk_scope(scope_body)
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
    }
    return sorted(names)


def lower_numpy_path_bindings(module):
    """Rewrite each qualifying name of the module and of its functions."""
    aliases = _numpy_aliases(module)
    if not aliases:
        return
    for name in _candidate_names(module.body):
        _lower_name(module, name, aliases)
    # Walked afterwards: a failed rewrite of the module replaces its statements.
    for scope in [n for n in ast.walk(module) if isinstance(n, _FUNCTIONS)]:
        for name in _candidate_names(scope.body):
            _lower_name(scope, name, aliases)
