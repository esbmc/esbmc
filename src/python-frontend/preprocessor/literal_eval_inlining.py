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

Code that does not compile is left alone: CPython raises SyntaxError for it.
"""

import ast

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
    arg = call.args[0]
    if _str_constant(arg):
        # eval() strips leading spaces and tabs; exec() does not.
        return arg.value.lstrip(" \t") if func == "eval" else arg.value
    if _builtin_call(arg, "compile", 3, sites) and all(map(_str_constant, arg.args)):
        return arg.args[0].value if arg.args[2].value == func else None
    return None


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

    def __init__(self, sites):
        self.sites = sites
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

    def visit_Expr(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        source = None if self.scopes else _literal_source(node.value, "exec", self.sites)
        body = None if source is None else _parse(source, "exec", node)
        if body is None:
            return self.generic_visit(node)
        return body or ast.copy_location(ast.Pass(), node)


def inline_literal_eval(module):
    """Replace each eval/exec of a string literal in `module` by its code."""
    _Inliner(binding_sites(module)).visit(module)
