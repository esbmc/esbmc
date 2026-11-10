import random

D = {"a": 1, "b": 2}


def f(k):
    return D[k]


f(random.choice(["z"]))
