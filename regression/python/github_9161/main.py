import ast
from ast import NodeVisitor as Visitor


class V(ast.NodeTransformer):
    pass


class W(Visitor):

    def count(self) -> int:
        return 1


def f(x: int) -> int:
    assert x != 3
    return x


f(2)
