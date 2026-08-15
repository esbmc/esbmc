import ast
from pytype_infer.dataflow_solver import *
from pytype_infer.lattice import *

def test_integer_constant_inference():
    expr = ast.parse("1", mode="eval").body

    context = InferenceContext(
        known_classes={}
    )

    result = infer_type_from_expr(
        expr,
        {},
        context,
    )

    assert isinstance(result, IntType)