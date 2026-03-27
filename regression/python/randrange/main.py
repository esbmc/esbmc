import random


def main():
    result: int = random.randrange(0, 10, 1)
    assert result >= 0 and result < 10

    result2: int = random.randrange(0, 20, 2)
    assert result2 >= 0 and result2 < 20
    assert result2 % 2 == 0

    result3: int = random.randrange(1, 10, 3)
    assert result3 >= 1 and result3 < 10
    assert (result3 - 1) % 3 == 0

    val1: int = random.randrange(2, 20, 3)
    assert val1 >= 2 and val1 < 20
    assert (val1 - 2) % 3 == 0

    val2: int = random.randrange(10, 0, -2)
    assert val2 <= 10 and val2 > 0
    assert (10 - val2) % 2 == 0

    try:
        val3: int = random.randrange(0, 10, 0)
        assert False  # should not reach here
    except ValueError:
        assert True

    result4: int = random.randrange(5, 6, 1)
    assert result4 == 5

    result5: int = random.randrange(10, 9, -1)
    assert result5 == 10

    try:
        val1: int = random.randrange(10, 0, 1)
        assert False  # should not reach here
    except ValueError:
        assert True

    try:
        val2: int = random.randrange(0, 10, -1)
        assert False  # should not reach here
    except ValueError:
        assert True

    try:
        val3: int = random.randrange(5, 5, 1)
        assert False  # should not reach here
    except ValueError:
        assert True


if __name__ == "__main__":
    main()
