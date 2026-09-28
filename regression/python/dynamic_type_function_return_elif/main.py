def f(cond1, cond2):
    if cond1:
        return 1
    elif cond2:
        return 2
    else:
        return "a"


x = f(nondet_bool(), nondet_bool())
assert x == 1 or x == 2 or x == "a"
