import math

calls = 0


def bump() -> float:
    global calls
    calls += 1
    return 0.5


y = math.exp(bump())
z = math.sin(bump())
assert calls == 2
assert y > 1.1 and z > 0.47
