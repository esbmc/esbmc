import ast


def first() -> ast.NoSuchClass:
    return None


assert first() is None
