# The assertions after a failing one hold, and the run still fails on it.
a = 2
b = nondet_int()
assert a * 2 == 4
assert b != 3
assert a + 1 == 3
assert a * a == 4
