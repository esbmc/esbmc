# #10331: abs() of the smallest int literal in the call argument is refused.
def func(x):
    if x > 0:
        return x
    else:
        return "neg"


b = func(-abs(-9223372036854775808))
assert b == "neg"
