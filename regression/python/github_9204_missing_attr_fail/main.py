import ast


def visit(node: ast.NoSuchClass) -> int:
    return 1


assert visit(None) == 1
