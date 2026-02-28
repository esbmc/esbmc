import random

x = random.randint(0, 100)


def foo(d: dict) -> dict:
    d["a"] = 2
    return d


d = {"a": x, "b": 2, "c": 3}

my_dict: dict = foo(d)

assert (my_dict["a"] >= 0 and my_dict["a"] <= 100)
