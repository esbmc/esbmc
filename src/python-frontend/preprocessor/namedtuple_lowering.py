"""Lower `collections.namedtuple` classes to plain tuples (#8487).

`N = namedtuple("N", fields)` at module level is dropped, a call `N(...)`
becomes the tuple of its arguments in field order, and `v.field` becomes
`v[i]` for a name `v` whose every binding in the module is `v = N(...)`.
Any other use of `N`, such as `isinstance(v, N)` or an annotation, is refused.
"""

import ast
import builtins
import keyword

from preprocessor.rejection import attach_location
# pylint: disable=invalid-name


def _refuse(message, node, module_name):
    exc = NotImplementedError(message)
    attach_location(exc, node, module_name)
    raise exc


def _namedtuple_aliases(module):
    """The names that call `collections.namedtuple` at module level."""
    names = set()
    for stmt in module.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "collections":
            names.update(a.asname or a.name for a in stmt.names if a.name == "namedtuple")
        elif isinstance(stmt, ast.Import):
            names.update(f"{a.asname or a.name}.namedtuple" for a in stmt.names
                         if a.name == "collections")
    return names


def _callee_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def _field_names(spec):
    """The field names a namedtuple field spec lists, or None if not literal."""
    if isinstance(spec, ast.Constant) and isinstance(spec.value, str):
        return spec.value.replace(",", " ").split()
    if isinstance(spec, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in spec.elts):
        return [e.value for e in spec.elts]
    return None


def _valid_fields(fields):
    return (fields is not None and len(set(fields)) == len(fields)
            and all(f.isidentifier() and not keyword.iskeyword(f) and not f.startswith("_")
                    for f in fields))


def _defaults(call, fields):
    """The constant defaults of the rightmost fields, or None if not supported."""
    if not call.keywords:
        return {}
    if len(call.keywords) != 1 or call.keywords[0].arg != "defaults":
        return None
    value = call.keywords[0].value
    if not (isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) <= len(fields)
            and all(isinstance(e, ast.Constant) for e in value.elts)):
        return None
    return dict(zip(fields[len(fields) - len(value.elts):], value.elts))


def _definition(stmt, aliases, module_name):
    """(name, fields, defaults) when @p stmt defines a namedtuple class."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)
            and _callee_name(stmt.value.func) in aliases):
        return None
    call = stmt.value
    fields = _field_names(call.args[1]) if len(call.args) == 2 else None
    defaults = _defaults(call, fields) if _valid_fields(fields) else None
    if defaults is None:
        _refuse("this form of namedtuple() is not supported", stmt, module_name)
    return stmt.targets[0].id, fields, defaults


def _names_bound_by(node):
    """The identifiers @p node binds or declares."""
    if isinstance(node, ast.Name):
        return [] if isinstance(node.ctx, ast.Load) else [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return node.names
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler,
                         ast.MatchAs, ast.MatchStar)):
        return [node.name] if node.name else []
    return []


def _bound_identifiers(module):
    """Per identifier, the nodes that bind it anywhere in @p module."""
    bound = {}
    for node in ast.walk(module):
        for name in _names_bound_by(node):
            bound.setdefault(name, []).append(node)
    return bound


def _assigned_value(target, parents):
    """The value assigned to @p target when it is the sole target of an assignment."""
    stmt = parents.get(id(target))
    if isinstance(stmt, ast.Assign) and stmt.targets == [target]:
        return stmt.value
    if isinstance(stmt, ast.AnnAssign) and stmt.target is target:
        return stmt.value
    return None


def _returns(node):
    """The return statements of the function @p node, outside nested scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Return):
            yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            yield from _returns(child)


def _returned_class(func, classes):
    """The class N when every return of @p func, which cannot fall off its end,
    is `return N(...)` (#8969), else None."""
    if not func.body or not isinstance(func.body[-1], ast.Return):
        return None
    returned = {
        r.value.func.id if isinstance(r.value, ast.Call) and isinstance(r.value.func, ast.Name)
        and r.value.func.id in classes else None
        for r in _returns(func)
    }
    return returned.pop() if len(returned) == 1 else None


def _constructed_class(site, parents, classes, functions):
    """The class N when @p site is the target of `name = N(...)`, or of a call
    to a function that always returns one, else None."""
    value = _assigned_value(site, parents) if isinstance(site, ast.Name) else None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in classes:
            return value.func.id
        func = functions.get(value.func.id)
        return _returned_class(func, classes) if func else None
    return None


def _has_star_import(module):
    return any(
        isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
        for n in ast.walk(module))


def _instance_names(module, classes, bound):
    """Per name whose every binding is `name = N(...)`, the class N."""
    if _has_star_import(module):
        return {}
    parents = {id(child): node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
    functions = {
        s.name: s
        for s in module.body if isinstance(s, ast.FunctionDef) and len(bound[s.name]) == 1
    }
    instances = {}
    for name, sites in bound.items():
        constructed = {_constructed_class(site, parents, classes, functions) for site in sites}
        if len(constructed) == 1 and None not in constructed and not hasattr(builtins, name):
            instances[name] = constructed.pop()
    return instances


def _class_attributes(module):
    """The attribute names some class of @p module may define."""
    names = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            names.add(node.attr)
        elif isinstance(node, ast.ClassDef):
            for inner in ast.walk(node):
                names.update(_names_bound_by(inner))
    return names


def _plain_arguments(node):
    return not any(isinstance(a, ast.Starred) for a in node.args) and all(k.arg is not None
                                                                          for k in node.keywords)


def _arguments(node, fields, defaults):
    """The arguments of the call @p node in field order, or None if invalid."""
    if len(node.args) > len(fields) or not _plain_arguments(node):
        return None
    given = dict(zip(fields, node.args))
    for kw in node.keywords:
        if kw.arg not in fields or kw.arg in given:
            return None
        given[kw.arg] = kw.value
    if any(f not in given and f not in defaults for f in fields):
        return None
    return [given[f] if f in given else ast.Constant(value=defaults[f].value) for f in fields]


class _Lowering(ast.NodeTransformer):

    def __init__(self, classes, instances, class_attributes, module_name):
        self.classes = classes
        self.instances = instances
        self.class_attributes = class_attributes
        self.module_name = module_name
        self.fields = {f for fields, _ in classes.values() for f in fields}

    def visit_Call(self, node):
        """The tuple a constructor call builds."""
        if not (isinstance(node.func, ast.Name) and node.func.id in self.classes):
            return self.generic_visit(node)
        node.args = [self.visit(a) for a in node.args]
        for kw in node.keywords:
            kw.value = self.visit(kw.value)
        fields, defaults = self.classes[node.func.id]
        elts = _arguments(node, fields, defaults)
        if elts is None:
            _refuse(f"this call to namedtuple class '{node.func.id}' is not supported", node,
                    self.module_name)
        return ast.copy_location(ast.Tuple(elts=elts, ctx=ast.Load()), node)

    def visit_Attribute(self, node):
        """The tuple element a field read selects."""
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in self.instances:
            # The value may be a namedtuple whose class is not known here; read
            # as a tuple it would raise a spurious AttributeError.
            if node.attr in self.fields and node.attr not in self.class_attributes:
                _refuse(
                    f"reading the namedtuple field '{node.attr}' of a value whose class "
                    "is not known is not supported", node, self.module_name)
            return node
        fields = self.classes[self.instances[node.value.id]][0]
        if node.attr not in fields:
            return node
        if not isinstance(node.ctx, ast.Load):
            _refuse(f"assigning to the namedtuple field '{node.attr}' is not supported", node,
                    self.module_name)
        index = ast.Constant(value=fields.index(node.attr))
        return ast.copy_location(ast.Subscript(value=node.value, slice=index, ctx=ast.Load()), node)

    def visit_Name(self, node):
        """Refuse a use of a namedtuple class other than a constructor call."""
        if node.id in self.classes:
            _refuse(f"namedtuple class '{node.id}' is supported only as a constructor", node,
                    self.module_name)
        return node


def lower_namedtuples(module, module_name):
    """Rewrite in place the namedtuple classes @p module defines at module level."""
    aliases = _namedtuple_aliases(module)
    if not aliases:
        return
    bound = _bound_identifiers(module)
    classes = {}
    body = []
    for stmt in module.body:
        found = _definition(stmt, aliases, module_name)
        if found is None:
            body.append(stmt)
            continue
        name, fields, defaults = found
        if len(bound[name]) != 1:
            _refuse(f"rebinding namedtuple class '{name}' is not supported", stmt, module_name)
        classes[name] = (fields, defaults)
    if not classes:
        return
    module.body = body
    instances = _instance_names(module, classes, bound)
    _Lowering(classes, instances, _class_attributes(module), module_name).visit(module)
    ast.fix_missing_locations(module)
