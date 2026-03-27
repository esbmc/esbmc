import math


def test_sqrt_float_literals_fail():
    assert math.sqrt(4.0) == 3.0  # wrong: sqrt(4.0) == 2.0, not 3.0


test_sqrt_float_literals_fail()
