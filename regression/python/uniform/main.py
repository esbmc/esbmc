import random


def main():
    result: float = random.uniform(0.0, 10.0)
    assert result >= 0.0 and result <= 10.0

    result2: float = random.uniform(-5.0, 5.0)
    assert result2 >= -5.0 and result2 <= 5.0

    result3: float = random.uniform(0.5, 1.5)
    assert result3 >= 0.5 and result3 <= 1.5

    x = random.uniform(1.0, 2.0)
    assert x >= 1.0
    assert x <= 2.0

    y = random.uniform(5.0, -1.0)
    assert y >= -1.0
    assert y <= 5.0

    result4: float = random.uniform(0.0, 10.0)
    assert 0.0 <= result4 <= 10.0

    result5: float = random.uniform(-5.0, 5.0)
    assert -5.0 <= result5 <= 5.0

    result6: float = random.uniform(1.0, 1.0)
    assert result6 == 1.0

    result7: float = random.uniform(5.0, 5.0)
    assert result7 == 5.0

    result8: float = random.uniform(10.0, 1.0)
    assert result8 >= 1.0 and result8 <= 10.0


if __name__ == "__main__":
    main()
