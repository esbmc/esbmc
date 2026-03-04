def test_from_import_inside_function_fail() -> None:
    from math import log10

    log10(-2.0)
    assert False


test_from_import_inside_function_fail()
