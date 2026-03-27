import random


def main():
    result: int = random.randrange(0, 10, 1)
    assert result == 10

    val1: int = random.randrange(2, 20, 3)
    assert val1 % 2 == 0

    val2: int = random.randrange(10, 0, -2)
    assert val2 >= 5


if __name__ == "__main__":
    main()
