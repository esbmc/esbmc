# The counterpart of range36-nondet: without the n >= 0 constraint the claim
# is false, because len(range(n)) is 0 for a negative n and not n.
n = nondet_int()
x = len(range(n))
assert x == n
