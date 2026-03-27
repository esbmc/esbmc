import math


def test_power_and_roots():
    assert 2**3 == 8
    assert math.pow(2, 3) == 8.0
    assert math.isclose(math.sqrt(16), 4.0)
    assert math.isclose(27**(1 / 3), 3.0)


test_power_and_roots()
