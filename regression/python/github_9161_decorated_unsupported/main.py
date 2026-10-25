import ast


class V(ast.NodeTransformer):

    @staticmethod
    def visit_Name(node):
        return node
