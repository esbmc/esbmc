import random

D = {"a": 1, "b": 2}


def f(k: str) -> int:
    return D[k]


assert f(random.choice(list(D.keys()))) > 0
