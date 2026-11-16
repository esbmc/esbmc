import ast


def visit(node: ast.Import, k: int) -> int:
    return k * 2


assert visit(None, 3) == 7
