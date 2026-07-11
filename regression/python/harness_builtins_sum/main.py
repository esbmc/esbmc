# Verification harness for sum (src/python-frontend/models/builtins.py).
#
# sum(l, start=0) returns start plus the sum of the list elements.
#
# REQUIRES:
#   R1: three non-deterministic integers, each bounded so the total stays
#       within the machine-int range and no arithmetic overflow occurs.
#
# ENSURES:
#   E1: sum(l) == a + b + c                     [accumulation is exact]
#   E2: sum(l, 10) == a + b + c + 10            [start offset is honoured]
a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

__ESBMC_assume(-1000 <= a <= 1000)
__ESBMC_assume(-1000 <= b <= 1000)
__ESBMC_assume(-1000 <= c <= 1000)

l: list[int] = [a, b, c]

assert sum(l) == a + b + c  # E1
assert sum(l, 10) == a + b + c + 10  # E2
