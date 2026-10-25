import ast


class Renamer(ast.NodeTransformer):

    def visit_Name(self, node):
        return node


def double(x: int) -> int:
    return 2 * x
