import math


def test_trigonometry():
    try:
        math.acos(2)
        assert False, "Expected ValueError"
    except ValueError:
        pass


test_trigonometry()
