# The sequence argument is not always a symbol: a call result has no element
# type recorded for it. That must keep the base model it had before the
# dispatch existed rather than become an error (#7673).
import random


def make() -> list[int]:
    return [1, 2]


def main() -> None:
    c = random.choice(make())
    assert c == 1 or c == 2


main()
