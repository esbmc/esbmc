"""Raise UnboundLocalError where a function reads a local it may not have bound.

A function-local read is checked against a definite-assignment analysis. A
name some read may reach unbound gets a flag, set at each of its bindings and
tested before that read. A name is left alone when the analysis cannot see all
of its bindings (walrus, except-as, del, import, nested def, match capture,
global/nonlocal) or when a nested scope references it.

Module-level code gets the same check, raising NameError, for a name bound only
by module-level defs, such as a call placed above the def of its function.
"""

import ast
import builtins
# pylint: disable=too-many-return-statements,too-few-public-methods

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp,
                  ast.SetComp, ast.DictComp, ast.GeneratorExp)
_FLAG_PREFIX = "__ESBMC_bound_"


def _own_scope_nodes(node):
    """Yield the nodes of a function body, not descending into nested scopes."""
    stack = list(reversed(node.body))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, _NESTED_SCOPES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(current))))


def _target_names(target):
    """Return the Name nodes an assignment target binds."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _target_names(elt)]
    return []


def _binding_targets(stmt):
    """Return the targets a statement binds by a form the analysis tracks."""
    if isinstance(stmt, ast.Assign):
        return stmt.targets
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        return [stmt.target]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in stmt.items if item.optional_vars]
    return []


def _bound_ids(stmt):
    """Return the names a statement binds (an AnnAssign without value binds none)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {stmt.name}
    if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
        return set()
    ids = set()
    for target in _binding_targets(stmt):
        ids.update(name.id for name in _target_names(target))
    return ids


def _untracked_names(node):
    """Return the names a node binds, or hides, in a way the analysis does not see."""
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    if isinstance(node, ast.ExceptHandler):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, _NESTED_SCOPES):
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        names.add(getattr(node, "name", None))
        return names
    return set()


def _candidates(func):
    """Return the locals whose every binding the analysis tracks."""
    if any(isinstance(node, ast.Match) for node in ast.walk(func)):
        return set()
    tracked, stores, excluded = set(), set(), set()
    for node in _own_scope_nodes(func):
        for target in _binding_targets(node):
            tracked.update(id(name) for name in _target_names(target))
        excluded |= _untracked_names(node)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores.add(node.id)
            if id(node) not in tracked or isinstance(node.ctx, ast.Del):
                excluded.add(node.id)
    args = func.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    params.update(a.arg for a in (args.vararg, args.kwarg) if a)
    return stores - excluded - params


def _unconditional_reads(expr, out):
    """Collect the names an expression always reads when it is evaluated."""
    if expr is None or isinstance(expr, _NESTED_SCOPES):
        return
    if isinstance(expr, ast.Name):
        if isinstance(expr.ctx, ast.Load):
            out.add(expr.id)
        return
    if isinstance(expr, ast.BoolOp):
        _unconditional_reads(expr.values[0], out)
        return
    if isinstance(expr, ast.IfExp):
        _unconditional_reads(expr.test, out)
        return
    for child in ast.iter_child_nodes(expr):
        _unconditional_reads(child, out)


def _head_exprs(stmt):
    """Return the expressions a statement evaluates on reaching it."""
    if isinstance(stmt, (ast.Expr, ast.Return)):
        return [stmt.value]
    if isinstance(stmt, ast.Assign):
        return [stmt.value] + stmt.targets
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return [stmt.value, stmt.target] if stmt.value else []
    if isinstance(stmt, (ast.If, ast.While, ast.Assert)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items]
    if isinstance(stmt, ast.Raise):
        return [stmt.exc, stmt.cause]
    return []


def _meet(states):
    """Intersect the reachable states; None when none is reachable."""
    reachable = [state for state in states if state is not None]
    return set.intersection(*reachable) if reachable else None


class _Analysis:
    """Definite assignment over one function body; bindings never unbind."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.unsafe_reads = {}

    def flow(self, stmts, bound):
        """Return the names bound after `stmts`, or None if the end is unreachable."""
        for stmt in stmts:
            if bound is None:
                return None
            reads = set()
            for expr in _head_exprs(stmt):
                _unconditional_reads(expr, reads)
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                reads.add(stmt.target.id)
            unsafe = (reads & self.candidates) - bound
            if unsafe:
                self.unsafe_reads[id(stmt)] = unsafe
            bound = self._effect(stmt, bound)
        return bound

    def _effect(self, stmt, bound):
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return None
        if isinstance(stmt, ast.If):
            return _meet([self.flow(stmt.body, bound), self.flow(stmt.orelse, bound)])
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            self.flow(stmt.body, bound | _bound_ids(stmt))
            self.flow(stmt.orelse, bound)
            return bound
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            # A context manager can suppress an exception raised in its body.
            self.flow(stmt.body, bound | _bound_ids(stmt))
            return bound | _bound_ids(stmt)
        if isinstance(stmt, ast.Try):
            ends = [self.flow(stmt.orelse, self.flow(stmt.body, bound))]
            ends.extend(self.flow(handler.body, bound) for handler in stmt.handlers)
            final_end = self.flow(stmt.finalbody, bound)
            try_end = _meet(ends)
            if try_end is None or final_end is None:
                return None
            return try_end | final_end
        return bound | _bound_ids(stmt)


def _flag_assign(name, value):
    return ast.Assign(targets=[ast.Name(id=_FLAG_PREFIX + name, ctx=ast.Store())],
                      value=ast.Constant(value=value))


def _local_error(name):
    return ("UnboundLocalError",
            f"cannot access local variable '{name}' where it is not associated with a value")


def _global_error(name):
    return ("NameError", f"name '{name}' is not defined")


def _guard(name, error):
    exc, message = error(name)
    return ast.If(
        test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=_FLAG_PREFIX + name, ctx=ast.Load())),
        body=[
            ast.Raise(exc=ast.Call(func=ast.Name(id=exc, ctx=ast.Load()),
                                   args=[ast.Constant(value=message)],
                                   keywords=[]),
                      cause=None)
        ],
        orelse=[],
    )


def _located(new_nodes, anchor):
    for new in new_nodes:
        ast.copy_location(new, anchor)
        ast.fix_missing_locations(new)
    return new_nodes


class _Rewriter:
    """Insert the flag updates and guards for the names in `flagged`."""

    def __init__(self, flagged, unsafe_reads, error):
        self.flagged = flagged
        self.unsafe_reads = unsafe_reads
        self.error = error

    def rewrite(self, stmts):
        """Return `stmts` with the guards before and the flag updates after."""
        out = []
        for stmt in stmts:
            guarded = sorted(self.unsafe_reads.get(id(stmt), set()) & self.flagged)
            out.extend(_located([_guard(name, self.error) for name in guarded], stmt))
            out.append(stmt)
            self._rewrite_children(stmt)
            bound = sorted(_bound_ids(stmt) & self.flagged)
            sets = _located([_flag_assign(name, True) for name in bound], stmt)
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                stmt.body[:0] = sets
            else:
                out.extend(sets)
        return out

    def _rewrite_children(self, stmt):
        if isinstance(stmt, _NESTED_SCOPES):
            return
        for field in ("body", "orelse", "finalbody"):
            block = getattr(stmt, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                setattr(stmt, field, self.rewrite(block))
        for handler in getattr(stmt, "handlers", []):
            handler.body = self.rewrite(handler.body)


def _guard_function(func):
    candidates = _candidates(func)
    if not candidates:
        return
    analysis = _Analysis(candidates)
    analysis.flow(func.body, set())
    flagged = set().union(*analysis.unsafe_reads.values())
    if not flagged:
        return
    body = _Rewriter(flagged, analysis.unsafe_reads, _local_error).rewrite(func.body)
    has_docstring = (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                     and isinstance(body[0].value.value, str))
    start = 1 if has_docstring else 0
    body[start:start] = _located([_flag_assign(name, False) for name in sorted(flagged)], func)
    func.body = body


def _module_candidates(module):
    """Return the globals bound only by module-level defs the analysis sees."""
    if any(isinstance(node, ast.Match) for node in ast.walk(module)):
        return set()
    defs, excluded = set(), set()
    for node in _own_scope_nodes(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            excluded.add(node.name)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            excluded.add(node.id)
        elif not isinstance(node, _NESTED_SCOPES):
            excluded |= _untracked_names(node)
    hidden = _names_bound_elsewhere(module)
    if hidden is None:
        return set()
    return {name for name in defs - excluded - hidden if not hasattr(builtins, name)}


def _names_bound_elsewhere(module):
    """Return the globals a nested scope may bind, or None if any name may be."""
    names = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.NamedExpr):
            names.add(node.target.id)
        elif isinstance(node, ast.Name) and node.id in ("globals", "vars", "exec", "eval"):
            return None
    return names


def _guard_module(module):
    candidates = _module_candidates(module)
    if not candidates:
        return
    analysis = _Analysis(candidates)
    analysis.flow(module.body, set())
    flagged = set().union(*analysis.unsafe_reads.values())
    if not flagged:
        return
    body = _Rewriter(flagged, analysis.unsafe_reads, _global_error).rewrite(module.body)
    module.body = _located([_flag_assign(name, False) for name in sorted(flagged)], body[0]) + body


def guard_unbound_locals(module):
    """Make every function in `module` raise UnboundLocalError on an unbound read,
    and its module-level code raise NameError on a read of a def not yet run."""
    _guard_module(module)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _guard_function(node)
