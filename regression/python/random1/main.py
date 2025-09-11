import random


def nondet_int():
    return random.choice([0, 1])  # Simulate non-deterministic choice


x: int = 90

while True:
    n: int = nondet_int()

    if n == 0 and x <= 100:
        x += 1
    elif n == 1 and x > 0:
        x -= 1

    assert 0 <= x <= 100, "Assertion failed: x is out of bounds!"
