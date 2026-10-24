import random


class Die:

    def roll(self) -> int:
        return 4


class Cup:

    def __init__(self) -> None:
        self.die = Die()


def f(random: Cup) -> int:
    return random.die.roll()


assert f(Cup()) == 4
