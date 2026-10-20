"""Lower `match` statements whose patterns need no destructuring to if/elif.

Literal and value patterns compare with `==`, `None`/`True`/`False` with `is`,
and wildcard, capture, `as` and `|` patterns combine these. A capture binds the
subject itself, so a guard reads the subject in place of the names its case
binds. A statement with any other pattern is left for the converter to refuse.
"""

import ast
import copy
# pylint: disable=invalid-name

_SUBJECT_PREFIX = "__esbmc_match_subject_"


def _bound_names(pattern):
    """The names @p pattern binds, or None if it needs more than a comparison."""
    if isinstance(pattern, (ast.MatchValue, ast.MatchSingleton)):
        return set()
    if isinstance(pattern, ast.MatchAs):
        inner = set() if pattern.pattern is None else _bound_names(pattern.pattern)
        if inner is None:
            return None
        return inner | ({pattern.name} if pattern.name else set())
    if isinstance(pattern, ast.MatchOr):
        alternatives = [_bound_names(p) for p in pattern.patterns]
        if any(names is None for names in alternatives):
            return None
        return set().union(*alternatives)
    return None


def _test(pattern, subject):
    """The condition under which @p pattern matches @p subject, or None if always."""
    if isinstance(pattern, ast.MatchValue):
        return ast.Compare(left=copy.deepcopy(subject),
                           ops=[ast.Eq()],
                           comparators=[copy.deepcopy(pattern.value)])
    if isinstance(pattern, ast.MatchSingleton):
        return ast.Compare(left=copy.deepcopy(subject),
                           ops=[ast.Is()],
                           comparators=[ast.Constant(value=pattern.value)])
    if isinstance(pattern, ast.MatchAs):
        return None if pattern.pattern is None else _test(pattern.pattern, subject)
    tests = [_test(p, subject) for p in pattern.patterns]
    if any(t is None for t in tests):
        return None
    return ast.BoolOp(op=ast.Or(), values=tests)


class _ReadSubject(ast.NodeTransformer):
    """Replace reads of the names a case binds with reads of the subject."""

    def __init__(self, names, subject):
        self.names = names
        self.subject = subject

    def visit_Name(self, node):
        """Read the subject where a bound name is read."""
        if node.id in self.names and isinstance(node.ctx, ast.Load):
            return copy.deepcopy(self.subject)
        return node


def _guard_is_rewritable(guard):
    # A nested scope or a walrus could rebind a captured name.
    return not any(
        isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                       ast.NamedExpr)) for n in ast.walk(guard))


def _is_lowerable(node, bound):
    return all(names is not None
               for names in bound) and all(case.guard is None or _guard_is_rewritable(case.guard)
                                           for case in node.cases)


def _case_branch(case, names, subject, orelse):
    """The statements running @p case when it matches and @p orelse otherwise."""
    conditions = [_test(case.pattern, subject)]
    if case.guard is not None:
        conditions.append(_ReadSubject(names, subject).visit(case.guard))
    conditions = [c for c in conditions if c is not None]
    bindings = [
        ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=copy.deepcopy(subject))
        for n in sorted(names)
    ]
    body = bindings + case.body
    if not conditions:
        return body
    test = conditions[0] if len(conditions) == 1 else ast.BoolOp(op=ast.And(), values=conditions)
    return [ast.If(test=test, body=body, orelse=orelse)]


class _MatchLowering(ast.NodeTransformer):

    def __init__(self):
        self.counter = 0

    def visit_Match(self, node):
        """An if/elif chain equivalent to @p node, or @p node if none is built."""
        self.generic_visit(node)
        bound = [_bound_names(case.pattern) for case in node.cases]
        if not _is_lowerable(node, bound):
            return node

        prefix = []
        subject = node.subject
        if not isinstance(subject, ast.Name):
            name = f"{_SUBJECT_PREFIX}{self.counter}"
            self.counter += 1
            prefix.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=subject))
            subject = ast.Name(id=name, ctx=ast.Load())

        orelse = []
        for case, names in reversed(list(zip(node.cases, bound))):
            orelse = _case_branch(case, names, subject, orelse)

        lowered = prefix + (orelse or [ast.Pass()])
        for stmt in lowered:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return lowered


def lower_match_statements(module):
    """Rewrite in place every `match` in @p module that _MatchLowering supports."""
    _MatchLowering().visit(module)
