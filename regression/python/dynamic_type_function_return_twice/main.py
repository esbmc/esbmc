def f(cond):
    if cond:
        return 1
    else:
        return "a"

a = f(nondet_bool())
b = f(nondet_bool())
assert a == 1 or a == "a"
assert b == 1 or b == "a"
