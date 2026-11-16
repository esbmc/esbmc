import ast
import json
from ast import Import


def visit(node: ast.Import) -> int:
    return 1


def decode(d: json.JSONDecoder, k: int) -> int:
    return k + 1


def quoted(node: "ast.Import") -> int:
    return 2


def imported(node: Import) -> int:
    return 3


class V(ast.NodeTransformer):

    def visit_Import(self, node: ast.Import) -> int:
        return 4


assert visit(None) == 1
assert decode(None, 2) == 3
assert quoted(None) == 2
assert imported(None) == 3
