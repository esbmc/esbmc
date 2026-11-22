import random


def f(flag):
    if flag:
        return 42
    return "unknown"


flag = random.random() < 0.5
c = f(flag)
assert c in (42, "unknown")
assert not (c not in (42, "unknown"))
assert (c in (1, "x")) == False
assert (c in [42, "unknown"])
if flag:
    v = 1
else:
    v = "a"
assert v in (1, "a")
assert v not in (2, "b")
