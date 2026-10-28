"""Refuse a read of an attribute only a skipped base `__init__` sets (#8844).

When a class defines `__init__` without calling its base's on every path, an
attribute that only the base `__init__` sets is missing on its instances, and
reading it raises AttributeError. The converter lays out one struct per class
with every inherited attribute, so the read gets a value instead. Such a
program is refused at the class's instantiation when the attribute is read
anywhere.
"""

import ast


def _self_stores(fn):
    """Attributes `fn` stores through its first parameter."""
    if not fn.args.args:
        return set()
    me = fn.args.args[0].arg
    return {
        n.attr
        for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
        and isinstance(n.value, ast.Name) and n.value.id == me
    }


def _init(cls):
    return next((s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"),
                None)


def _is_init_call(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "__init__")


def _returns(stmt):
    return not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
        isinstance(n, ast.Return) for n in ast.walk(stmt))


def _always_calls_an_init(body):
    """Whether `body` calls an `__init__` or raises before it can return, on
    every path (#9308)."""
    for stmt in body:
        if _is_init_call(stmt) or isinstance(stmt, ast.Raise):
            return True
        if (isinstance(stmt, ast.If) and _always_calls_an_init(stmt.body)
                and _always_calls_an_init(stmt.orelse)):
            return True
        if _returns(stmt):
            return False
    return False


def _ancestors(cls, classes):
    """The module's classes `cls` derives from, or None when one is defined
    elsewhere."""
    found = []
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "object":
            continue
        if not isinstance(base, ast.Name) or base.id not in classes:
            return None
        parent = classes[base.id]
        above = _ancestors(parent, classes)
        if above is None:
            return None
        found += [parent] + above
    return found


def _missing_attributes(cls, classes, outside_stores):
    """Attributes only a base `__init__` that `cls` skips would set."""
    init = _init(cls)
    ancestors = _ancestors(cls, classes)
    if init is None or _always_calls_an_init(init.body) or not ancestors:
        return set()
    from_bases = set()
    for parent in ancestors:
        if _init(parent) is not None:
            from_bases |= _self_stores(_init(parent))
    own = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            own |= _self_stores(stmt)
        own |= {
            t.id
            for t in ast.walk(stmt) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
        }
    return from_bases - own - outside_stores


def _attribute_uses(module, classes):
    """Attributes stored outside every class's `__init__`, and attributes read."""
    inits = {id(n) for c in classes.values() for n in ast.walk(_init(c) or ast.Pass())}
    outside_stores, reads = set(), set()
    for n in ast.walk(module):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            reads.add(n.attr)
        elif isinstance(n, ast.Attribute) and id(n) not in inits:
            outside_stores.add(n.attr)
    return outside_stores, reads


def refuse_skipped_base_init(module, module_name):
    """Raise NotImplementedError at the instantiation of such a class."""
    classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
    if len(classes) < 2:
        return
    outside_stores, reads = _attribute_uses(module, classes)
    for name, cls in classes.items():
        missing = _missing_attributes(cls, classes, outside_stores) & reads
        calls = [
            n for n in ast.walk(module)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        ]
        if not missing or not calls:
            continue
        attr = sorted(missing)[0]
        exc = NotImplementedError(
            f"unsupported: '{name}.__init__' does not call the base __init__ that sets "
            f"'{attr}' on every path, so reading '{attr}' on a '{name}' instance raises "
            "AttributeError, which is not modelled")
        exc.esbmc_location = (module_name, calls[0].lineno, calls[0].col_offset)
        raise exc
