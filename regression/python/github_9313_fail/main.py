import ast


def first() -> ast.Import:
    node: ast.Import = None
    return node


class Visitor:

    def __init__(self, count: int) -> None:
        self.last: ast.AST = None
        self.count = count


top: ast.Import = first()
v = Visitor(2)
assert top is None
assert v.last is None and v.count == 3
