def test_rounding() -> None:
    assert round(3.6) == 4
    assert round(3.14159, 2) == 3.14
    assert round(-3.6) == -4
    assert round(-2.5) == -2
    assert round(0.5) == 0
    assert round(1.5) == 2
    assert round(2.5) == 2


test_rounding()


git add src/python-frontend/function_call_expr.cpp \
        src/python-frontend/function_call_expr.h

gc "[python-frontend] Fix round() semantics and IR correctness #3638"


git add src/python-frontend/python_consteval.cpp

gc "[python-frontend] Complete round() constant folding #3638"


git add src/python-frontend/type_utils.h

gc "[python-frontend] Remove round from is_builtin_type #3638"

git add regression/python/github_3638/

gc "[python-frontend] test expand round() regression tests #3638"
