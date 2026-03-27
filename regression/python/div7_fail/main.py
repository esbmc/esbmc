import random as rand

cond: int = rand.random()


def div1(x: int) -> int:
    if (not cond):
        return 42 // x
    else:
        return x // 10


x: int = rand.random()

assert div1(x) != 1
