import ast


def is_import(node: ast.Import) -> bool:
    return True


assert is_import(ast.Import(names=[]))
