# #10331: the call is evaluated when the str return is at top level.
def func(x):
    if x > 0:
        return x
    return "neg"


b = func(-abs(nondet_int()))
assert b == 0
