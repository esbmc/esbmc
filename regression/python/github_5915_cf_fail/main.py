# Soundness guard for #5915 (control-flow-reassigned power base).
# n is 2 or 3 depending on the nondet branch, so (n + 0) ** 2 is 4 or 9.
# The old fold read n's stale symbol-table value (3) through the +,-,*,/ tree
# and proved `== 9` even on the n == 2 path -- a false negative. The base must
# stay symbolic, so this assertion must be VIOLATED.
c = nondet_int()
n = 2
if c > 0:
    n = 3
assert (n + 0) ** 2 == 9
