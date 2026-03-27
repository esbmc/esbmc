import math


def test_factorial_and_combinatorics():
    assert math.factorial(0) == 1
    assert math.factorial(5) == 120
    assert math.comb(5, 2) == 10
    assert math.perm(5, 2) == 20

    # Edge cases
    try:
        math.factorial(-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


test_factorial_and_combinatorics()
