import math


def test_acos_success() -> None:
    assert math.acos(1.0) == 0.0
    assert math.acos(0.0) >= 1.0
    assert math.acos(0.0) <= 2.0


test_acos_success()
