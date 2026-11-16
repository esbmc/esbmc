def visit(node: ast.Import) -> int:
    return 1


import ast

assert visit(None) == 1
