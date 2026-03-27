import random


def main():
    result: float = random.uniform(0.0, 10.0)
    assert result >= 10.0

    z = random.uniform(0.0, 1.0)
    assert z < 0.0
    assert z > 1.0


if __name__ == "__main__":
    main()
