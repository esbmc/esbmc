import ast
from pytype_infer.dataflow_solver import *
from pytype_infer.lattice import *

def test_integer_literal_inference():
    expr = ast.parse("2", mode="eval").body

    context = InferenceContext(
        known_classes={}
    )

    result = infer_type_from_expr(
        expr,
        {},
        context,
    )

    assert isinstance(result, IntType)