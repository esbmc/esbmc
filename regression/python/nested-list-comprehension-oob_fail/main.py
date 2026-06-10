# The element-type clamp in handle_index_access (#5129) must not mask a genuine
# out-of-bounds access: a constant outer index past the end of a comprehension
# still has its value read at the real index, so the runtime list-bounds check
# must fire (out-of-bounds read in list) rather than silently passing or
# crashing the SMT backend.

C = [[0.0 for _ in range(2)] for _ in range(2)]
x = C[9][0]
assert x - 0.0 == 0.0
