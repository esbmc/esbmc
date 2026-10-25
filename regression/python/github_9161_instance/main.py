import ast


class V(ast.NodeTransformer):
    pass


class W(V):

    def count(self) -> int:
        return 1


w = W()
assert w.count() == 1
