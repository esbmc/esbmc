def mk() -> str:
    return "a"

cond = nondet_bool()
if cond:
    x = 1
else:
    x = mk()
if not cond:
    assert x == "a"
