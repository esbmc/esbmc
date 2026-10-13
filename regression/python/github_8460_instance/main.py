import random


class Die:

    def roll(self) -> int:
        return 4


def f(random: Die) -> int:
    return random.roll()


assert f(Die()) == 4
