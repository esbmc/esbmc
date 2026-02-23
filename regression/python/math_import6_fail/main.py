def test_import_inside_function_fail() -> None:
    import math

    math.atanh(2.0)
    assert False


test_import_inside_function_fail()
