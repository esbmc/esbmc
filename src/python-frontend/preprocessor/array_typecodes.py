"""Refuse an array.array the array model does not handle.

The model stores ints only, so a float or unicode array, or one whose
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


def reject_unmodelled_array_typecodes(module, module_name):
    """Raise for the first array.array call the model cannot represent."""
    modules, imported = _array_bindings(module, module_name)
    if not modules and not imported:
        return
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not _is_array_call(node, modules, imported):
            continue
        typecode = node.args[0] if node.args else None
        if (not isinstance(typecode, ast.Constant) or not isinstance(typecode.value, str)
                or typecode.value in ("f", "d", "u", "w")):
            _refuse(node, module_name, "array.array is modelled only for an integer typecode")
