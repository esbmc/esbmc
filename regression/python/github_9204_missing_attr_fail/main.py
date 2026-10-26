# #9204: CPython raises AttributeError evaluating this annotation.
import ast


def f(node: ast.NoSuchClass) -> int:
    return 1


assert f(None) == 1
