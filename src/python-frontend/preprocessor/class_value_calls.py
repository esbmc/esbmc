"""Construct through a module-level name bound to one of several classes (#8478).

The converter recognises a constructor call only by the class's own name. For a
name of the entry module whose every binding is a module-level `alias = A` or a
conditional of such classes, such as `alias = A if c else B`, and whose every
read is a call `alias(...)` at module level after a statement that binds it on
every path, each binding instead stores the index of the class it binds, and
each call becomes the construction of the class that index names. The receiver
checks then see which class each path constructs.

A name bound the same way to one of several modules, each bound solely by a
top-level `import`, is read through its attributes instead: each `alias.attr`
becomes the attribute of the module the index names (#9808).
"""

import ast
import copy

from preprocessor.binding_sites import binding_sites
from preprocessor.receiver_classes import _scope_nodes, _select
from preprocessor.unbound_locals import _located

_PREFIX = "ESBMC_class_value_"
_MODULE_PREFIX = "ESBMC_module_value_"


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


def _used_name(node, attributes):
    """The Name `node` uses as a callee, or as the object of an attribute read
    when `attributes`, else None."""
    if not attributes and isinstance(node, ast.Call):
        return node.func
    if attributes and isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.value
    return None


def _reads_are_late_uses(module, name, attributes):
    """Whether every read of `name` is a use (see `_used_name`) at module level
    in a top-level statement after one that binds `name` on every path; CPython
    raises NameError for a read that runs before any binding."""
    first = next((i for i, stmt in enumerate(module.body) if _binds(stmt, name)), None)
    if first is None:
        return False
    uses = set()
    allowed = set()
    for stmt in module.body[first + 1:]:
        for node in _scope_nodes([stmt]):
            uses.add(id(_used_name(node, attributes)))
            allowed.add(id(node))
    return all(
        id(node) in uses and id(node) in allowed for node in ast.walk(module)
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


class _AttributeRewriter(ast.NodeTransformer):
    """Replace each attribute read of an alias, and each call of one, by that
    of the module its index selects; the converter cannot call a conditional."""

    def __init__(self, aliases):
        self.aliases = aliases

    def _alias_of(self, node):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in self.aliases):
            return self.aliases[node.value.id]
        return None

    def _selected(self, node, attribute, rebuild):
        selector, ordered = self._alias_of(attribute)
        options = [
            rebuild(
                ast.Attribute(value=ast.Name(id=m, ctx=ast.Load()),
                              attr=attribute.attr,
                              ctx=ast.Load())) for m in ordered
        ]
        return _located([_select(selector, options)], node)[0]

    def visit_Call(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        if self._alias_of(node.func) is None:
            return self.generic_visit(node)
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        return self._selected(
            node, node.func, lambda f: ast.Call(
                func=f, args=copy.deepcopy(node.args), keywords=copy.deepcopy(node.keywords)))

    def visit_Attribute(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        if self._alias_of(node) is None:
            return self.generic_visit(node)
        return self._selected(node, node, lambda f: f)


def _selected_aliases(module, values, attributes, replaced):
    """The aliases of `module` bound to one of several `values` and read as
    `_used_name` describes, each mapped to its selector and the values in index
    order; records in `replaced` the new target and value of their bindings."""
    sites = binding_sites(module)
    aliases = {}
    for name, assigns in _alias_bindings(module, values).items():
        leaves = {v for a in assigns for v in _class_leaves(a.value, values)}
        if (len(leaves) < 2 or sites[name] != len(assigns)
                or not _reads_are_late_uses(module, name, attributes)):
            continue
        ordered = sorted(leaves)
        index = {v: number for number, v in enumerate(ordered)}
        selector = (_MODULE_PREFIX if attributes else _PREFIX) + name
        aliases[name] = (selector, ordered)
        for a in assigns:
            replaced[id(a)] = (selector, _class_codes(a.value, index))
    return aliases


def _imported_modules(module, sites):
    """The names a top-level `import` binds and nothing else does."""
    return {
        a.asname or a.name.split(".")[0]
        for stmt in module.body if isinstance(stmt, ast.Import) for a in stmt.names
        if sites[a.asname or a.name.split(".")[0]] == 1
    }


def dispatch_class_value_calls(module):
    """Rewrite the bindings and uses of each alias of the entry `module` that
    qualifies, as the module docstring describes."""
    sites = binding_sites(module)
    classes = {s.name for s in module.body if isinstance(s, ast.ClassDef) and sites[s.name] == 1}
    replaced = {}
    aliases = _selected_aliases(module, classes, False, replaced)
    module_aliases = _selected_aliases(module, _imported_modules(module, sites), True, replaced)
    if not replaced:
        return
    for node in _scope_nodes(module.body):
        if id(node) in replaced:
            selector, codes = replaced[id(node)]
            node.targets = [ast.Name(id=selector, ctx=ast.Store())]
            node.value = codes
            ast.fix_missing_locations(node)
    _CallRewriter(aliases).visit(module)
    _AttributeRewriter(module_aliases).visit(module)
