import ast


class V(ast.NodeTransformer):

    def visit_Name(self, node, limit=1 // 0):
        return node
