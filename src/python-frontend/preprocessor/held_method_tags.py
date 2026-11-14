"""Tag the methods an attribute holds when they return different types (#10365).

`self.a = self.f` on one path and `self.a = self.g` on another, with `f`
returning a str and `g` an int, leaves `self.a()` one return type in the
converter, which refuses the call. When every store into an attribute named
`a` is such a bound method of one class, and each method it holds returns a
str, an int or a bool, each method is marked to return a tagged value and each
call `obj.a(...)` to give one, as function_value_types does for a call through
a function value. The converter checks at each call that the method called is
one of the marked ones. Any other store, a subclass of the class or a method
it cannot type keeps the refusal.
"""

import ast

from preprocessor.function_value_types import _TAGGABLE, _Analysis, _differ

_HOOKS = {"setattr", "delattr", "getattr", "vars", "__dict__", "__setattr__", "__getattr__"}


def _plain_methods(cls):
    """The undecorated methods `cls` defines once each, by name, or None for a class with
    bases, decorators or keywords."""
    if cls.bases or cls.keywords or cls.decorator_list:
        return None
    defs = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [m.name for m in defs]
    return {
        m.name: m
        for m in defs
        if isinstance(m, ast.FunctionDef) and not m.decorator_list and names.count(m.name) == 1
    }


def _self_method(store, method, methods):
    """The method of `methods` that `store`, a statement of `method`, binds as
    `self.a = self.m`, else None."""
    if not (isinstance(store, ast.Assign) and len(store.targets) == 1 and method.args.args):
        return None
    self_name = method.args.args[0].arg
    value, target = store.value, store.targets[0]
    if not (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
            and value.value.id == self_name and isinstance(target.value, ast.Name)
            and target.value.id == self_name):
        return None
    return methods.get(value.attr)


def _held(module, methods):
    """`attr -> methods` for each attribute whose every store or delete in `module` is
    `self.attr = self.m` in one of `methods`."""
    parents = {id(c): n for n in ast.walk(module) for c in ast.iter_child_nodes(n)}
    owner = {id(n): m for m in methods.values() for n in ast.walk(m)}
    held = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Attribute) or isinstance(node.ctx, ast.Load):
            continue
        store = parents.get(id(node))
        method = owner.get(id(store))
        stored = method and _self_method(store, method, methods)
        if stored and held.get(node.attr, []) is not None:
            held.setdefault(node.attr, []).append(stored)
        else:
            held[node.attr] = None
    return {attr: ms for attr, ms in held.items() if ms}


def _kinds(analysis, held):
    """The return types of `held`, when each is in `_TAGGABLE` and two differ, else None."""
    kinds = [analysis.def_kind(m) for m in held]
    if not all(k in _TAGGABLE for k in kinds):
        return None
    return kinds if any(_differ(a, b) for a in kinds for b in kinds) else None


def _mark(module, attr, held):
    for method in held:
        method.esbmc_tagged_return = True
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func,
                                                     ast.Attribute) and node.func.attr == attr:
            node.esbmc_tagged_call = True
    module.esbmc_tagged_returns = True


def _reflective(module):
    names = {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(module) if isinstance(n, ast.Attribute)}
    return bool(names & _HOOKS)


def _unsubclassed(classes):
    """The classes of `classes` that none of them derives from."""
    bases = {b.id for c in classes for b in c.bases if isinstance(b, ast.Name)}
    return [c for c in classes if c.name not in bases]


def tag_held_methods(module):
    """Mark the methods and the calls of the attributes that hold methods of different
    return types, when every condition above holds."""
    if _reflective(module):
        return
    analysis = None
    for cls in _unsubclassed([s for s in module.body if isinstance(s, ast.ClassDef)]):
        methods = _plain_methods(cls) or {}
        for attr, held in _held(module, methods).items():
            analysis = analysis or _Analysis(module, "")
            if attr not in methods and _kinds(analysis, held):
                _mark(module, attr, held)
