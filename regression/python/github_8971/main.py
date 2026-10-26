class Box:
    def __init__(self, v):
        self.v = v


use_int = nondet_bool()
if use_int:
    x = nondet_int()
else:
    x = "text"
b = Box(x)
assert isinstance(b.v, int) == use_int
assert isinstance(b.v, str) == (not use_int)
assert b.v == x
