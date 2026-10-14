def make_int() -> int:
    return 7


x = 0
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

if cond:
    x = make_int()
    assert x == 999
