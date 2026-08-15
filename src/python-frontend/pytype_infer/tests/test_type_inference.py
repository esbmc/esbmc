import ast
from pytype_infer.dataflow_solver import *
from pytype_infer.lattice import *


def test_negative_integer_inference():
    expr = ast.parse("-1", mode="eval").body

    print(ast.dump(expr, indent=2))

    context = InferenceContext(
        known_classes={}
    )

    result = infer_type_from_expr(
        expr,
        {},
        context,
    )

    print("result", result)
    print("result", type(result))
    assert isinstance(result, IntType)