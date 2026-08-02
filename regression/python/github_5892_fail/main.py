# github #5892 (negative): the tuple membership assume admits every listed
# literal, so `side` may legitimately be 'b'. Asserting `side == 'a'` alone must
# therefore FAIL. This also proves the positive test is non-vacuous: the assume
# is satisfiable with a value other than 'a'.
side = nondet_str()
__ESBMC_assume(len(side) <= 8)
__ESBMC_assume(side in ('a', 'b'))
assert side == 'a'
