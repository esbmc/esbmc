"""Construct through a module-level name bound to one of several classes (#8478).

The converter recognises a constructor call only by the class's own name. For a
name of the entry module whose every binding is a module-level `alias = A` or a
conditional of such classes, such as `alias = A if c else B`, and whose every
read is a call `alias(...)`, or an `is`, `is not`, `==` or `!=` comparison with
one of those classes, at module level after a statement that binds it on every
path, each binding instead stores the index of the class it binds, each call
becomes the construction of the class that index names, and each comparison a
test of that index. The receiver checks then see which class each path
constructs.

A name bound the same way to one of several modules, each bound solely by a
top-level `import`, is read through its attributes instead: each `alias.attr`
becomes the attribute of the module the index names (#9808).

A name bound to a function or a module, such as `f = sub.g` on one path and
`f = sub` on the other, and only called, has each call become the call of what
its index names; calling a module raises TypeError (#10232).
"""

import ast
import copy

from preprocessor.binding_sites import binding_sites
from preprocessor.receiver_classes import _scope_nodes, _select
from preprocessor.unbound_locals import _located

_PREFIX = "ESBMC_class_value_"
_MODULE_PREFIX = "ESBMC_module_value_"


def _leaf_key(value):
    """`name` for a Name and `name.attr` for an attribute of a Name, else None."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        return value.value.id + "." + value.attr
    return None


def _leaf_expr(key):
    """The expression `_leaf_key` gives `key` for."""
    return ast.parse(key, mode="eval").body


def _class_leaves(value, classes):
    """The classes the arms of `value` name, or None when an arm is anything
    but one of `classes`."""
    if isinstance(value, ast.IfExp):
        body, orelse = _class_leaves(value.body, classes), _class_leaves(value.orelse, classes)
        return None if body is None or orelse is None else body + orelse
    key = _leaf_key(value)
    return [key] if key in classes else None


def _class_codes(value, index):
    """`value` with each class it names replaced by its index; the tests keep
    their order and are evaluated once."""
    if isinstance(value, ast.IfExp):
        return ast.IfExp(test=value.test,
                         body=_class_codes(value.body, index),
                         orelse=_class_codes(value.orelse, index))
    return ast.Constant(value=index[_leaf_key(value)])


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


_IDENTITY_OPS = (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)


def _compared_class(node, name, classes):
    """The class `node` compares `name` with, when `node` is `name is C`,
    `C is name` or the like with `is not`, `==` or `!=`, for C in `classes`."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], _IDENTITY_OPS)):
        return None
    sides = [node.left, node.comparators[0]]
    if not all(isinstance(side, ast.Name) for side in sides):
        return None
    ids = [side.id for side in sides]
    if ids[0] == name and ids[1] in classes:
        return ids[1]
    if ids[1] == name and ids[0] in classes:
        return ids[0]
    return None


def _used_names(node, name, values, attributes):
    """The Names `node` uses `name` through: the object of an attribute read
    when `attributes`, else a callee or a side of a comparison
    `_compared_class` accepts."""
    if attributes:
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            return [node.value]
        return []
    if isinstance(node, ast.Call):
        return [node.func]
    if _compared_class(node, name, values) is not None:
        return [node.left, node.comparators[0]]
    return []


def _reads_are_late_uses(module, name, values, attributes):
    """Whether every read of `name` is a use `_used_names` accepts at module
    level in a top-level statement after one that binds `name` on every path;
    CPython raises NameError for a read that runs before any binding."""
    first = next((i for i, stmt in enumerate(module.body) if _binds(stmt, name)), None)
    if first is None:
        return False
    callees = set()
    allowed = set()
    for stmt in module.body[first + 1:]:
        for node in _scope_nodes([stmt]):
            callees.update(id(used) for used in _used_names(node, name, values, attributes))
            allowed.add(id(node))
    return all(
        id(node) in callees and id(node) in allowed for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name)


class _CallRewriter(ast.NodeTransformer):
    """Replace each call of an alias by the construction its index selects,
    and each comparison of an alias with a class by a test of its index."""

    def __init__(self, aliases, classes):
        self.aliases = aliases
        self.classes = classes

    def visit_Compare(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        for name, (selector, ordered) in self.aliases.items():
            cls = _compared_class(node, name, self.classes)
            if cls is None:
                continue
            # A class the alias never binds has no index and so never matches.
            number = ordered.index(cls) if cls in ordered else -1
            same = isinstance(node.ops[0], (ast.Is, ast.Eq))
            test = ast.Compare(left=ast.Name(id=selector, ctx=ast.Load()),
                               ops=[ast.Eq() if same else ast.NotEq()],
                               comparators=[ast.Constant(value=number)])
            return _located([test], node)[0]
        return node

    def visit_Call(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in self.aliases):
            return node
        selector, ordered = self.aliases[node.func.id]
        constructions = [
            ast.Call(func=_leaf_expr(cls),
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


def _selected_aliases(module, values, attributes, replaced, called_among=None):
    """The aliases of `module` bound to one of several `values` and read as
    `_used_names` describes, each mapped to its selector and the values in index
    order; records in `replaced` the new target and value of their bindings.
    With `called_among`, an alias is taken only when one of its values is in
    it, and every read of it must be a call."""
    sites = binding_sites(module)
    aliases = {}
    for name, assigns in _alias_bindings(module, values).items():
        leaves = {v for a in assigns for v in _class_leaves(a.value, values)}
        if called_among is not None and not leaves & called_among:
            continue
        compared = values if called_among is None else ()
        if (len(leaves) < 2 or sites[name] != len(assigns) or id(assigns[0]) in replaced
                or not _reads_are_late_uses(module, name, compared, attributes)):
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


def _imported_names(module, sites):
    """The names a top-level `import` or `from ... import` binds and nothing
    else does."""
    return {
        a.asname or a.name.split(".")[0]
        for stmt in module.body if isinstance(stmt, (ast.Import, ast.ImportFrom))
        for a in stmt.names if a.name != "*" and sites[a.asname or a.name.split(".")[0]] == 1
    }


def _callable_values(module, sites, imported):
    """The names of `imported` and of functions defined once at module level,
    and each `name.attr` read of an `imported` name that nothing assigns."""
    stored = {
        node.value.id
        for node in ast.walk(module) if isinstance(node, ast.Attribute)
        and not isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
    }
    values = set(imported)
    values.update(s.name for s in module.body
                  if isinstance(s, ast.FunctionDef) and sites[s.name] == 1)
    values.update(
        _leaf_key(node) for node in ast.walk(module)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id in imported and node.value.id not in stored)
    return values


def dispatch_class_value_calls(module):
    """Rewrite the bindings and uses of each alias of the entry `module` that
    qualifies, as the module docstring describes."""
    sites = binding_sites(module)
    classes = {s.name for s in module.body if isinstance(s, ast.ClassDef) and sites[s.name] == 1}
    replaced = {}
    aliases = _selected_aliases(module, classes, False, replaced)
    module_aliases = _selected_aliases(module, _imported_modules(module, sites), True, replaced)
    imported = _imported_names(module, sites)
    # Only calls: `sub.g` and a `g` imported from `pkg.sub` are one function
    # under two keys, so comparing indices would be wrong.
    callables = _selected_aliases(module, _callable_values(module, sites, imported), False,
                                  replaced, imported)
    if not replaced:
        return
    for node in _scope_nodes(module.body):
        if id(node) in replaced:
            selector, codes = replaced[id(node)]
            node.targets = [ast.Name(id=selector, ctx=ast.Store())]
            node.value = codes
            ast.fix_missing_locations(node)
    _CallRewriter(aliases, classes).visit(module)
    _CallRewriter(callables, set()).visit(module)
    _AttributeRewriter(module_aliases).visit(module)
