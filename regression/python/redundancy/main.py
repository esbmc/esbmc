import random


def redundancy_examples(x: int, a: int, b: int, c: int):
    # (1) Defensive programming redundancy
    # Multiple equivalent constraints instead of a single on
    assert (x >= 0 and x >= 0 and x > -1) == (x >= 0)

    # (2) Variable explosion redundancy
    # Unnecessary temporaries instead of direct computation
    tmp1 = a + b
    tmp2 = tmp1 * c
    result_redundant = tmp2
    result_direct = (a + b) * c
    assert result_redundant == result_direct

    # (3) Implied constraints redundancy
    # Two overlapping conditions where one is enough
    assert (x < 10 and x <= 9) == (x <= 9)


x = random.randint(1, 1000)
a = random.randint(1, 1000)
b = random.randint(1, 1000)
c = random.randint(1, 1000)
redundancy_examples(x, a, b, c)
