# Control: both arms are ints, so the expression is left as it is.
n = abs(nondet_int()) % 3
c = nondet_bool()
k = n if c else abs(n)
assert isinstance(k, int)
assert k == n
