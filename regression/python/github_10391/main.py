# 10391: a str returned beside None compares by its characters.
import random


def pick(flag):
    if flag:
        return "ab"
    return None


v = pick(random.random() < 0.5)
assert v is None or v == "ab"
if v is not None:
    assert v != "cd"
