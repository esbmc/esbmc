import math

def test_fractional_power():
    assert math.isclose(8 ** (1/3), 2.0)
    assert math.isclose(27 ** (1/3), 3.0)

test_fractional_power()
