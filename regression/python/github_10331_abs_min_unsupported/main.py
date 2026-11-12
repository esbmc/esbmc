# #10331: a computed -2**63 in the call argument is still refused.
def func(x):
    if x > 0:
        return x
    else:
        return "neg"


b = func(-abs(nondet_int() - 1))
assert b == "neg"
