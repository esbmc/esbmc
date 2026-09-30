cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

if isinstance(x, int):
    y = x.bit_length()
    assert y != 1
