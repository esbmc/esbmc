# #9204: a parameter annotated with a class of a module with no model.
import ast
import json
from ast import Import


def f(node: ast.Import) -> int:
    return 1


def g(d: json.JSONDecoder, node: "ast.Import", *rest: Import, k: Import = None) -> int:
    return 2


class V(ast.NodeTransformer):

    def visit_Import(self, node: ast.Import) -> int:
        return 3


assert f(None) == 1
assert g(None, None) == 2
