import math


def main() -> None:
    # math.dist([0, 0], [3, 4]) is 5.0, not 4.0.
    assert math.dist([0, 0], [3, 4]) == 4.0


main()
