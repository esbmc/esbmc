# The same rebind works for a class instance and for a tuple, and a later
# scalar rebind still lands in a slot of its own.


class P:
    def __init__(self, v: int) -> None:
        self.v: int = v


cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = P(7)
assert x.v == 7

if cond:
    y = 1
else:
    y = "a"
y = (1, 2)
assert y[0] == 1

if cond:
    z = 1
else:
    z = "a"
z = [1, 2]
z = 5
assert z == 5
