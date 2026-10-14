from typing import Callable, Any

def f(v):
    return v == 1

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

g: Callable[[Any], bool]
g = f
assert g(x)
