cond = nondet_bool()
if cond:
    x = "a"
else:
    x = 1

if isinstance(x, int):
    y = x.bit_length()
    assert y != 1
