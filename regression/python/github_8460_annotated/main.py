import random


def f(random: str) -> int:
    return len(random.split())


assert f("a b") == 2
