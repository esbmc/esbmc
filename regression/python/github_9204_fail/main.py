# #9204: a parameter annotated with a class of a module with no model.
import ast


def f(node: ast.Import) -> int:
    return node.lineno


assert f(None) == 1
