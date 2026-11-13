# x is tagged from call-site divergence; one branch derives y from x, the
# other just aliases x, and both still agree on y's type at the join.
def f(x):
    if nondet_bool():
        y = x + 1
    else:
        y = x
    return y


a = f(3)
b = f("s")
