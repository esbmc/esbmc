def f(cond):
    if cond:
        return 1
    return "a"


x = f(nondet_bool())
assert x == 1 or x == "a"
