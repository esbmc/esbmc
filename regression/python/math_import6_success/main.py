import math


def test_import_inside_function() -> None:

    assert math.hypot(3.0, 4.0) == 5.0
    assert math.ulp(1.0) > 0.0


test_import_inside_function()
