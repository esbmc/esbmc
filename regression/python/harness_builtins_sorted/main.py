# Verification harness for sorted (src/python-frontend/models/builtins.py).
#
# sorted(l) returns a new list with the same elements in non-decreasing order.
#
# REQUIRES:
#   R1: three fully non-deterministic integers, covering every ordering.
#
# ENSURES (for s = sorted(l)):
#   E1: len(s) == len(l)                        [length preserved]
#   E2: s[0] <= s[1] <= s[2]                     [output is non-decreasing]
#   E3: s[0] == min(l)                           [smallest element first]
#   E4: s[2] == max(l)                           [largest element last]
#   E5: s[0] + s[1] + s[2] == a + b + c          [multiset sum preserved:
#       a lightweight permutation check — no element is dropped or duplicated]
a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

__ESBMC_assume(-1000 <= a <= 1000)
__ESBMC_assume(-1000 <= b <= 1000)
__ESBMC_assume(-1000 <= c <= 1000)

l: list[int] = [a, b, c]
s: list[int] = sorted(l)

assert len(s) == 3  # E1
assert s[0] <= s[1]  # E2
assert s[1] <= s[2]  # E2
assert s[0] == min(l)  # E3
assert s[2] == max(l)  # E4
assert s[0] + s[1] + s[2] == a + b + c  # E5
