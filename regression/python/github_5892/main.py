# github #5892: `x in (str_literal, ...)` must be per-element equality, not a
# substring (strstr) search. With a symbolic string constrained to be a member
# of the tuple, `x == a or x == b` must hold. Older builds lowered this to
# strstr and unsoundly accepted the empty string.
side = nondet_str()
__ESBMC_assume(len(side) <= 8)
__ESBMC_assume(side in ('a', 'b'))
assert side == 'a' or side == 'b'
