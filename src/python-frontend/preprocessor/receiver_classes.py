"""Refuse an attribute read on a name whose class depends on the path (#8408).

A name holds one C type, the class it is first bound to. When it is bound to
instances of different classes, a method call or attribute read on it would be
resolved against that one class on every path.
"""

import ast

from preprocessor.rejection import attach_location


def _constructed_classes(value, classes, returns):
    """The classes `value` constructs, directly or through a function
    annotated to return one, or None when it is anything else."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in classes:
            return {value.func.id}
        if value.func.id in returns:
            return {returns[value.func.id]}
    return None


def _defining_class(cls, attr, classes, seen=frozenset()):
    """The class whose body defines `attr` as a method, searching the bases
    depth first; None when no class of this module defines it. A method no
    class defines is missing from every class alike."""
    if cls in seen:
        return None
    node = classes[cls]
    if any(isinstance(s, ast.FunctionDef) and s.name == attr for s in node.body):
        return cls
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in classes:
            found = _defining_class(base.id, attr, classes, seen | {cls})
            if found is not None:
                return found
    return None


def _scope_nodes(body):
    """The nodes of a scope, without descending into nested scopes."""
    stack = list(body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child,
                              (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def _receiver_classes(body, classes, returns):
    """Per name bound in this scope to instances of more than one class of
    this module, the set of those classes."""
    bound = {}
    for node in _scope_nodes(body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        constructed = _constructed_classes(value, classes, returns)
        if constructed is not None:
            bound.setdefault(target.id, set()).update(constructed)
    return {name: found for name, found in bound.items() if len(found) > 1}


def _check_scope(body, classes, returns, module_name):
    receivers = _receiver_classes(body, classes, returns)
    if not receivers:
        return
    called = set()
    for node in _scope_nodes(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(id(node.func))
    for node in _scope_nodes(body):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in receivers):
            continue
        found = receivers[node.value.id]
        defining = {_defining_class(c, node.attr, classes) for c in found}
        if id(node) in called and len(defining) == 1:
            continue
        names = " or ".join(sorted(found))
        exc = NotImplementedError(
            f"'{node.value.id}.{node.attr}', where '{node.value.id}' holds an instance of "
            f"{names} depending on the path, is not supported")
        attach_location(exc, node, module_name)
        raise exc


def refuse_path_dependent_receivers(module, module_name):
    classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
    if len(classes) < 2:
        return
    returns = {
        s.name: s.returns.id
        for s in module.body if isinstance(s, ast.FunctionDef) and isinstance(s.returns, ast.Name)
        and s.returns.id in classes
    }
    _check_scope(module.body, classes, returns, module_name)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_scope(node.body, classes, returns, module_name)
