import ast


class V(ast.NoSuchClass):
    pass


def f(x: int) -> int:
    assert x != 3
    return x


f(4)
