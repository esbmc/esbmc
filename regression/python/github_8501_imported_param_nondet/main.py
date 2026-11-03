import lib

f = nondet_float()
__ESBMC_assume(f == 1.5)
assert lib.neg(f) == -1.5
b = nondet_bool()
assert lib.flip(b) != b
s = nondet_str()
assert lib.ident(s) == s
