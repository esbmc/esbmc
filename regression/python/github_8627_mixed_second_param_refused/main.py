# The calls that pass an int as a still disagree on b, which one copy per
# type of a cannot type.
def op(a, b):
    return a + b


r1 = op(1, 1)
r2 = op(1, 1.5)
r3 = op(1.5, 1)
assert r2 == 2.5
