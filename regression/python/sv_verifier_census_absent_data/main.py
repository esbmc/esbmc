import _sv_verifier

n = _sv_verifier.nondet_int()
b = _sv_verifier.nondet_bool()
assert n == n
assert b or not b
