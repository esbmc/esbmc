def greet(v):
    return "Hello " + v


cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

if isinstance(x, str):
    assert greet(x) == "Hello a"
