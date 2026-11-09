"""Drop the parameter and return annotations that name numpy.ndarray.

`np.ndarray` names neither a shape nor a dtype, so it says no more than no
annotation; without one the annotator takes the type from the calls and the
returns, as it does for an unannotated array.
"""

import ast

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _ndarray_names(module):
    """The names an import of the module binds to numpy and to numpy.ndarray."""
    modules, classes = set(), set()
    for stmt in module.body:
        if isinstance(stmt, ast.Import):
            modules |= {a.asname or a.name for a in stmt.names if a.name == "numpy"}
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "numpy":
            classes |= {a.asname or a.name for a in stmt.names if a.name == "ndarray"}
    return modules, classes


def _names_ndarray(annotation, modules, classes):
    if isinstance(annotation, ast.Name):
        return annotation.id in classes
    return (isinstance(annotation, ast.Attribute) and annotation.attr == "ndarray"
            and isinstance(annotation.value, ast.Name) and annotation.value.id in modules)


def drop_ndarray_annotations(module):
    """Remove each annotation of a function of the module that names ndarray."""
    modules, classes = _ndarray_names(module)
    if not modules and not classes:
        return
    for function in [n for n in ast.walk(module) if isinstance(n, _FUNCTIONS)]:
        if _names_ndarray(function.returns, modules, classes):
            function.returns = None
        arguments = function.args
        for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
            if _names_ndarray(arg.annotation, modules, classes):
                arg.annotation = None
