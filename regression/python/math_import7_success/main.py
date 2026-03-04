from math import erf, erfc


def test_from_import_inside_function() -> None:

    assert erf(1.0) > 0.8
    assert erfc(1.0) < 0.2


test_from_import_inside_function()
