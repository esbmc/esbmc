import random
import m
import n

calls = 0


def arg() -> int:
    global calls
    calls += 1
    return calls


if random.random() < 0.5:
    x = m
else:
    x = n
assert x.V == 10 or x.V == 20
assert x.add(arg()) == x.V // 10 + 1
assert calls == 1
