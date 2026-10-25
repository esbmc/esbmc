from ast import NodeTransformer as NT


class V(NT):
    """Never instantiated, so its unmodelled base is never needed."""

    def visit_Name(self, node, flag=False):
        return node


def f(x: int) -> int:
    assert x != 3
    return x


f(4)
