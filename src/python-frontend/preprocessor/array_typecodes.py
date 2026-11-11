"""Route array.array calls to the model class for their typecode.

The model has an int class and a double class, so a "d" array is routed to
the double class, and a single-precision float or unicode array, or one whose
typecode is not a literal, is refused here rather than reaching the model.
"""

import ast

from preprocessor.rejection import attach_location


def _refuse(node, module_name, detail):
    exc = NotImplementedError(f"unsupported: {detail}")
    attach_location(exc, node, module_name)
    raise exc


def _array_bindings(module, module_name):
    """Names bound to the array module, and whether array.array is imported."""
    modules, imported = set(), False
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "array")
        elif isinstance(node, ast.ImportFrom) and node.module == "array":
            for alias in node.names:
                if alias.name == "array" and alias.asname not in (None, "array"):
                    # A model class called under another name is not resolved.
                    _refuse(node, module_name, "importing array.array under another name")
                imported = imported or alias.name == "array"
    return modules, imported


def _is_array_call(call, modules, imported):
    func = call.func
    if isinstance(func, ast.Name):
        return imported and func.id == "array"
    return (isinstance(func, ast.Attribute) and func.attr == "array"
            and isinstance(func.value, ast.Name) and func.value.id in modules)


_DOUBLE_ARRAY = "_double_array"


def _import_double_array(module):
    """Bind the double class wherever array.array is imported by name."""
    for node in ast.walk(module):
        if (isinstance(node, ast.ImportFrom) and node.module == "array"
                and all(a.name != _DOUBLE_ARRAY for a in node.names)):
            node.names.append(ast.alias(name=_DOUBLE_ARRAY))


def _refuse_class_values(module, module_name, modules, imported):
    """A "d" array is not an instance of the int class, so any use of
    array.array other than a call (isinstance, type(x) is, a base class)
    could answer differently from CPython."""
    called = {id(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)}
    for node in ast.walk(module):
        if (isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in called
                and isinstance(node.ctx, ast.Load)
                and _is_array_call(ast.Call(func=node), modules, imported)):
            _refuse(node, module_name, "using array.array as a value alongside a 'd' array")


def route_array_typecodes(module, module_name):
    """Send each "d" array.array call to the double class; raise for the
    first call the model cannot represent."""
    modules, imported = _array_bindings(module, module_name)
    if not modules and not imported:
        return
    routed = False
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not _is_array_call(node, modules, imported):
            continue
        typecode = node.args[0] if node.args else None
        if (not isinstance(typecode, ast.Constant) or not isinstance(typecode.value, str)
                or typecode.value in ("f", "u", "w")):
            _refuse(node, module_name,
                    "array.array is modelled only for an integer or 'd' typecode")
        if typecode.value == "d":
            if isinstance(node.func, ast.Name):
                node.func.id = _DOUBLE_ARRAY
            else:
                node.func.attr = _DOUBLE_ARRAY
            routed = True
    if not routed:
        return
    _refuse_class_values(module, module_name, modules, imported)
    if imported:
        _import_double_array(module)
