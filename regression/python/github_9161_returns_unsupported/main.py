import ast


class V(ast.NodeTransformer):

    def visit_Name(self, node) -> ast.AST:
        return node
