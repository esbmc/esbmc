import ast
from pytype_infer.dataflow_solver import *
from pytype_infer.lattice import *
from annotate_ast import type_to_ast_annotation
from annotate_ast import *

# def test_integer_constant_inference():
#     expr = ast.parse("1", mode="eval").body

#     context = InferenceContext(
#         known_classes={}
#     )

#     result = infer_type_from_expr(
#         expr,
#         {},
#         context,
#     )

#     assert isinstance(result, IntType)

def test(x: float, y: float) -> float:
    return x * y

assert test(2.0, 3.0) == 6.0    

def test2(p: list[float]) -> float:
    x = p[0]
    return x * x

p = [3.0, 4.0]
assert test2(p) == 9.0

def test_type_to_ast_annotation_primitives():
    assert ast.unparse(
        type_to_ast_annotation(IntType())
    ) == "int"

    assert ast.unparse(
        type_to_ast_annotation(FloatType())
    ) == "float"

    assert ast.unparse(
        type_to_ast_annotation(BoolType())
    ) == "bool"

    assert ast.unparse(
        type_to_ast_annotation(StrType())
    ) == "str"

node = type_to_ast_annotation(NoneType())
print(ast.dump(node))
print(ast.unparse(node))    