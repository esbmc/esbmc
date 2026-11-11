"""Inline eval() and exec() of a string literal as the code it spells (#9805).

`eval(s)` and `exec(s)`, or the same of `compile(s, name, mode)` in the
matching mode, run code fixed before the program runs, so it is converted in
place of the call. A call is inlined only in a module that never binds the
names involved, and only where the inlined code resolves names as CPython
does:

- `exec` only as a statement of the module body, where its code runs in the
  module's namespace.
- `eval` at module or class level, or in a function nested in no other
  function or comprehension. In a function, the expression may hold no
  nested scope, walrus or `super`, which see a different scope once inlined.

`exec(name)` is inlined the same way when the module binds `name` only by
module-level assignments of such strings (#10230). Each string's code runs
under a test that `name` holds it, so the read of `name` stays in place, and
the call itself is kept for any other value.

Code that does not compile is left alone: CPython raises SyntaxError for it.
"""

import ast
from collections import Counter, defaultdict

from preprocessor.binding_sites import binding_sites

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_NESTED_SCOPES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _builtin_call(node, name, nargs, sites):
    """Whether `node` calls the builtin `name` with `nargs` positional args."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            and not sites[name] and not node.keywords and len(node.args) == nargs)


def _str_constant(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _literal_source(call, func, sites):
    """The string `call` passes to the builtin `func`, else None."""
    if not _builtin_call(call, func, 1, sites):
        return None
    return _code_source(call.args[0], func, sites)


def _code_source(arg, func, sites):
    """The code `arg` gives the builtin `func`, when fixed before the run."""
    if _str_constant(arg):
        # eval() strips leading spaces and tabs; exec() does not.
        return arg.value.lstrip(" \t") if func == "eval" else arg.value
    if _builtin_call(arg, "compile", 3, sites) and all(map(_str_constant, arg.args)):
        return arg.args[0].value if arg.args[2].value == func else None
    return None


def _module_statements(body):
    """The statements of a module `body`, at any depth outside a def or class."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, _FUNCTIONS + (ast.ClassDef, )):
            continue
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, (ast.ExceptHandler, ast.match_case)):
                yield from _module_statements(child.body)
        for field in ("body", "orelse", "finalbody"):
            yield from _module_statements(getattr(stmt, field, []))


def _reads(node):
    """How many times `node` reads each name."""
    return Counter(n.id for n in ast.walk(node)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))


def _executed_names(module, sites):
    """How many statements `exec(name)` `module` holds for each name."""
    return Counter(n.value.args[0].id for n in ast.walk(module)
                   if isinstance(n, ast.Expr) and _builtin_call(n.value, "exec", 1, sites)
                   and isinstance(n.value.args[0], ast.Name))


def _code_names(sources):
    """The names the code of `sources`, (source, mode) pairs, binds, and
    those it reads or binds."""
    binds, seen = Counter(), Counter()
    for source, mode in sources:
        try:
            tree = ast.parse(source, mode=mode)
        except (SyntaxError, ValueError):
            continue
        binds.update(binding_sites(tree))
        seen.update(_reads(tree))
    seen.update(binds)
    return binds, seen


def _held_strings(module, sites):
    """For each name `exec(name)` runs, the module-level assignments of code
    `exec` takes to it, with that code, in order."""
    executed = _executed_names(module, sites)
    held = defaultdict(list)
    for stmt in _module_statements(module.body) if executed else ():
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in executed):
            source = _code_source(stmt.value, "exec", sites)
            if source is not None:
                held[stmt.targets[0].id].append((source, stmt))
    return dict(held)


def _inlined_sources(module, sites, held):
    """The code, as (source, mode) pairs, that inlining may place in `module`."""
    sources = [(s, "exec") for assigns in held.values() for s, _ in assigns]
    for func in ("exec", "eval"):
        sources += [(s, func) for s in (_literal_source(n, func, sites) for n in ast.walk(module))
                    if s is not None]
    return sources


def _compiles_as_string(module, name, assigns, sites, seen):
    """Whether the code objects `assigns` bind to `name` behave as the strings
    they compile: they compile, `name` is bound nowhere else, and only
    statements `exec(name)` read it."""
    reads = _reads(module)
    return (len(assigns) == sites[name] and not seen[name]
            and not any(reads[h] or sites[h] for h in ("globals", "vars", "locals"))
            and reads[name] == _executed_names(module, sites)[name]
            and all(_parse(s, "exec", stmt) is not None for s, stmt in assigns))


def _parse(source, mode, location):
    """The body `source` parses to in `mode`, placed at `location`, else None."""
    try:
        compile(source, "<string>", mode, dont_inherit=True)
    except (SyntaxError, ValueError):
        return None
    tree = ast.parse(source, mode=mode)
    for node in ast.walk(tree):
        if hasattr(node, "lineno"):
            ast.copy_location(node, location)
    return tree.body


def _sees_enclosing_function(expr):
    """Whether `expr` reads a scope that differs once inlined in a function."""
    return any(
        isinstance(n, _NESTED_SCOPES + (ast.NamedExpr, )) or (
            isinstance(n, ast.Name) and n.id in ("super", "__class__")) for n in ast.walk(expr))


class _Inliner(ast.NodeTransformer):

    def __init__(self, sites, held):
        self.sites = sites
        self.held = held
        self.scopes = []

    def _visit_scope(self, node):
        self.scopes.append(node)
        self.generic_visit(node)
        self.scopes.pop()
        return node

    visit_FunctionDef = _visit_scope  # noqa: N815
    visit_AsyncFunctionDef = _visit_scope  # noqa: N815
    visit_ClassDef = _visit_scope  # noqa: N815
    visit_Lambda = _visit_scope  # noqa: N815
    visit_ListComp = _visit_scope  # noqa: N815
    visit_SetComp = _visit_scope  # noqa: N815
    visit_DictComp = _visit_scope  # noqa: N815
    visit_GeneratorExp = _visit_scope  # noqa: N815

    def _eval_allowed(self, expr):
        enclosing = [s for s in self.scopes if not isinstance(s, ast.ClassDef)]
        if not enclosing:
            return True
        return (len(enclosing) == 1 and isinstance(self.scopes[-1], _FUNCTIONS)
                and not _sees_enclosing_function(expr))

    def visit_Call(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        source = _literal_source(node, "eval", self.sites)
        expr = None if source is None else _parse(source, "eval", node)
        if expr is None or not self._eval_allowed(expr):
            return node
        return expr

    def _exec_held(self, node):
        """`exec(name)` as a test of each string `name` may hold, else None."""
        if self.scopes or not _builtin_call(node.value, "exec", 1, self.sites):
            return None
        arg = node.value.args[0]
        assigns = self.held.get(arg.id) if isinstance(arg, ast.Name) else None
        if not assigns:
            return None
        sources = list(dict.fromkeys(s for s, _ in assigns))
        bodies = [_parse(s, "exec", node) for s in sources]
        if None in bodies:
            return None
        # A value no binding gives, e.g. one a star import stores, still
        # reaches the call itself.
        chain = [node]
        for source, body in reversed(list(zip(sources, bodies))):
            test = ast.Compare(left=ast.Name(id=arg.id, ctx=ast.Load()),
                               ops=[ast.Eq()],
                               comparators=[ast.Constant(value=source)])
            chain = [ast.If(test=test, body=body or [ast.Pass()], orelse=chain)]
        return [ast.fix_missing_locations(ast.copy_location(chain[0], node))]

    def visit_Expr(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        held = self._exec_held(node)
        if held is not None:
            return held
        source = None if self.scopes else _literal_source(node.value, "exec", self.sites)
        body = None if source is None else _parse(source, "exec", node)
        if body is None:
            return self.generic_visit(node)
        return body or ast.copy_location(ast.Pass(), node)


def inline_literal_eval(module):
    """Replace each eval/exec of a string literal in `module` by its code."""
    sites = binding_sites(module)
    held = _held_strings(module, sites)
    binds, seen = _code_names(_inlined_sources(module, sites, held))
    # Inlined code that rebinds a builtin would change what a later call runs.
    if any(binds[f] for f in ("exec", "eval", "compile")):
        return
    for name, assigns in list(held.items()):
        if all(_str_constant(stmt.value) for _, stmt in assigns):
            continue
        if not _compiles_as_string(module, name, assigns, sites, seen):
            del held[name]
            continue
        for source, stmt in assigns:
            stmt.value = ast.copy_location(ast.Constant(value=source), stmt.value)
    _Inliner(sites, held).visit(module)
