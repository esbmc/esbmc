# -2**53 - 1 is not rounded to -2**53 through an Optional (esbmc/esbmc#10390).
import random


def big(x: int):
    if x < 0:
        return -9007199254740993
    return None


x = random.randint(-20, -1)
w = big(x)
assert w == -9007199254740992
