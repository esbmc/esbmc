class V(ast.NodeTransformer):
    pass


import ast


def f(x: int) -> int:
    assert x != 3
    return x


f(4)
