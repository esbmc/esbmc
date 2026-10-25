import ast


def f(x: int) -> int:
    assert x != 3
    return x


class V(ast.NodeTransformer):
    limit = f(4)
