"""Refuse an attribute read or isinstance on a name whose class depends on the
path (#8408, #8407).

A name holds one C type, the class it is first bound to. When it is bound to
instances of different classes, a method call, attribute read or isinstance on
it would be resolved against that one class on every path.
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
    if isinstance(value, ast.IfExp):
        arms = [
            set() if isinstance(arm, ast.Constant) and arm.value is None else _constructed_classes(
                arm, classes, returns) for arm in (value.body, value.orelse)
        ]
        if None not in arms and any(arms):
            return arms[0] | arms[1]
    return None


def _is_subclass(cls, name, classes, seen=frozenset()):
    """Whether class `cls` of this module is `name` or derives from it."""
    if cls == name:
        return True
    if cls in seen:
        return False
    return any(
        isinstance(base, ast.Name) and (base.id == name or base.id in classes
                                        and _is_subclass(base.id, name, classes, seen | {cls}))
        for base in classes[cls].bases)


def _has_outside_base(cls, classes, seen=frozenset()):
    """Whether a base of class `cls`, direct or inherited, is not a class of
    this module (other than object), so its ancestry is not known here."""
    if cls in seen:
        return False
    for base in classes[cls].bases:
        if not isinstance(base, ast.Name):
            return True
        if base.id in classes:
            if _has_outside_base(base.id, classes, seen | {cls}):
                return True
        elif base.id != "object":
            return True
    return False


def _classinfo_names(node):
    """The names an isinstance classinfo lists, or None for any other form."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple) and all(isinstance(e, ast.Name) for e in node.elts):
        return [e.id for e in node.elts]
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


def _isinstance_answer(cls, classinfo, classes):
    """isinstance's answer for an instance of class `cls`, or None when it is
    not known here."""
    names = _classinfo_names(classinfo)
    if names is None:
        return None
    if not all(n in classes or n == "object" for n in names) and _has_outside_base(cls, classes):
        return None
    return any(n == "object" or _is_subclass(cls, n, classes) for n in names)


def _header_nodes(stmt):
    """The nodes of `stmt` outside its nested statements and scopes."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(child for child in ast.iter_child_nodes(node)
                     if not isinstance(child, (ast.stmt, ast.Lambda)))


def _isinstance_receiver(node, receivers):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance" and len(node.args) == 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id in receivers):
        return node.args[0].id
    return None


def _scope_blocks(body):
    yield body
    for node in _scope_nodes(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield value


def _single_class(stmt, classes, returns):
    """The name `stmt` binds and the one class it constructs, or None."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target, value = stmt.target, stmt.value
    else:
        return None
    constructed = _constructed_classes(value, classes, returns)
    if isinstance(target, ast.Name) and constructed is not None and len(constructed) == 1:
        return target.id, next(iter(constructed))
    return None


def _definite_classes(body, receivers, classes, returns):
    """Per isinstance call on a receiver, the class a constructor bound the
    receiver to earlier in the same block, with no rebinding in between."""
    result = {}
    for block in _scope_blocks(body):
        current = {}
        for stmt in block:
            for node in _header_nodes(stmt):
                name = _isinstance_receiver(node, receivers)
                if name in current:
                    result[id(node)] = current[name]
            for node in _scope_nodes([stmt]):
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    current.pop(node.id, None)
            bound = _single_class(stmt, classes, returns)
            if bound is not None:
                current[bound[0]] = bound[1]
    return result


def _static_classes(body, receivers, classes, returns):
    """Per receiver, the class of its first binding in source order, which
    gives the name its C type; None when that binding is not a constructor."""
    constructs = {}
    first = {}
    for node in _scope_nodes(body):
        bound = _single_class(node, classes, returns) if isinstance(node, ast.stmt) else None
        if bound is not None:
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            constructs[id(target)] = bound[1]
        if (isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
                and node.id in receivers):
            key = (node.lineno, node.col_offset)
            if node.id not in first or key < first[node.id][0]:
                first[node.id] = (key, node)
    return {name: constructs.get(id(node)) for name, (_, node) in first.items()}


def _check_isinstance(node, receivers, classes, context, module_name):
    """Refuse isinstance on a path-dependent receiver unless its answer is
    known: the check reads the receiver's one static class."""
    name = _isinstance_receiver(node, receivers)
    if name is None:
        return
    definite, static = context
    found = receivers[name]
    answers = {_isinstance_answer(c, node.args[1], classes) for c in found}
    if len(answers) == 1 and None not in answers:
        return
    if id(node) in definite and static.get(name) is not None:
        here = _isinstance_answer(definite[id(node)], node.args[1], classes)
        if here is not None and here == _isinstance_answer(static[name], node.args[1], classes):
            return
    exc = NotImplementedError(
        f"isinstance() on '{name}', which holds an instance of {' or '.join(sorted(found))} "
        f"depending on the path, is not supported")
    attach_location(exc, node, module_name)
    raise exc


def _isinstance_context(body, receivers, classes, returns, params):
    """The definite class at each isinstance call and each receiver's static
    class, without a parameter or a global, which take their C type from
    outside this scope."""
    outside = set(params).union(*(node.names for node in _scope_nodes(body)
                                  if isinstance(node, (ast.Global, ast.Nonlocal))))
    static = _static_classes(body, receivers, classes, returns)
    return (_definite_classes(body, receivers, classes, returns), {
        name: cls
        for name, cls in static.items() if name not in outside
    })


def _parameters(func):
    args = func.args
    return {
        a.arg
        for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
        if a is not None
    }


def _check_scope(body, classes, returns, module_name, params=frozenset()):
    receivers = _receiver_classes(body, classes, returns)
    if not receivers:
        return
    context = _isinstance_context(body, receivers, classes, returns, params)
    called = set()
    for node in _scope_nodes(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(id(node.func))
    for node in _scope_nodes(body):
        _check_isinstance(node, receivers, classes, context, module_name)
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
            _check_scope(node.body, classes, returns, module_name, _parameters(node))
