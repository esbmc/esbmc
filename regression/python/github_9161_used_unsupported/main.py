import ast


class V(ast.NodeTransformer):
    pass


def f(x: int) -> int:
    assert x != 3
    return x


v = V()
f(4)
