"""Construct through a module-level name bound to one of several classes (#8478).

The converter recognises a constructor call only by the class's own name. For a
name of the entry module whose every binding is a module-level `alias = A` or a
conditional of such classes, such as `alias = A if c else B`, and whose every
read is a call `alias(...)` at module level after a statement that binds it on
every path, each binding instead stores the index of the class it binds, and
each call becomes the construction of the class that index names. The receiver
checks then see which class each path constructs.
"""

import ast
import copy

from preprocessor.binding_sites import binding_sites
from preprocessor.receiver_classes import _scope_nodes, _select
from preprocessor.unbound_locals import _located

_PREFIX = "ESBMC_class_value_"


def _class_leaves(value, classes):
    """The classes the arms of `value` name, or None when an arm is anything
    but one of `classes`."""
    if isinstance(value, ast.IfExp):
        body, orelse = _class_leaves(value.body, classes), _class_leaves(value.orelse, classes)
        return None if body is None or orelse is None else body + orelse
    if isinstance(value, ast.Name) and value.id in classes:
        return [value.id]
    return None


def _class_codes(value, index):
    """`value` with each class it names replaced by its index; the tests keep
    their order and are evaluated once."""
    if isinstance(value, ast.IfExp):
        return ast.IfExp(test=value.test,
                         body=_class_codes(value.body, index),
                         orelse=_class_codes(value.orelse, index))
    return ast.Constant(value=index[value.id])


def _alias_bindings(module, classes):
    """Per name, the module-scope assignments `name = <classes>` binding it."""
    found = {}
    for node in _scope_nodes(module.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _class_leaves(node.value, classes) is not None):
            found.setdefault(node.targets[0].id, []).append(node)
    return found


def _binds(stmt, name):
    """Whether `stmt` binds `name` on every path through it."""
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    if isinstance(stmt, ast.If):
        return all(any(_binds(s, name) for s in arm) for arm in (stmt.body, stmt.orelse))
    return False


def _reads_are_late_calls(module, name):
    """Whether every read of `name` is the callee of a call at module level in
    a top-level statement after one that binds `name` on every path; CPython
    raises NameError for a call that runs before any binding."""
    first = next((i for i, stmt in enumerate(module.body) if _binds(stmt, name)), None)
    if first is None:
        return False
    callees = set()
    allowed = set()
    for stmt in module.body[first + 1:]:
        for node in _scope_nodes([stmt]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callees.add(id(node.func))
            allowed.add(id(node))
    return all(
        id(node) in callees and id(node) in allowed for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name)


class _CallRewriter(ast.NodeTransformer):
    """Replace each call of an alias by the construction its index selects."""

    def __init__(self, aliases):
        self.aliases = aliases

    def visit_Call(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in self.aliases):
            return node
        selector, ordered = self.aliases[node.func.id]
        constructions = [
            ast.Call(func=ast.Name(id=cls, ctx=ast.Load()),
                     args=copy.deepcopy(node.args),
                     keywords=copy.deepcopy(node.keywords)) for cls in ordered
        ]
        return _located([_select(selector, constructions)], node)[0]


def dispatch_class_value_calls(module):
    """Rewrite the bindings and calls of each alias of the entry `module` that
    qualifies, as the module docstring describes."""
    sites = binding_sites(module)
    classes = {s.name for s in module.body if isinstance(s, ast.ClassDef) and sites[s.name] == 1}
    aliases = {}
    replaced = {}
    for name, assigns in _alias_bindings(module, classes).items():
        leaves = {cls for a in assigns for cls in _class_leaves(a.value, classes)}
        if (len(leaves) < 2 or sites[name] != len(assigns)
                or not _reads_are_late_calls(module, name)):
            continue
        ordered = sorted(leaves)
        index = {cls: number for number, cls in enumerate(ordered)}
        selector = _PREFIX + name
        aliases[name] = (selector, ordered)
        for a in assigns:
            replaced[id(a)] = (selector, _class_codes(a.value, index))
    if not aliases:
        return
    for node in _scope_nodes(module.body):
        if id(node) in replaced:
            selector, codes = replaced[id(node)]
            node.targets = [ast.Name(id=selector, ctx=ast.Store())]
            node.value = codes
            ast.fix_missing_locations(node)
    _CallRewriter(aliases).visit(module)
