def f(cond):
    if cond:
        return 1
    else:
        return "a"

x = f(nondet_bool())
assert x == 2
