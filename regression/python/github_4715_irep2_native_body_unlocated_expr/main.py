# A bare `print(expr)` whose argument is compound arrives as a code_expression2t
# with no location. The native arm needs one for the OTHER it emits, so it
# declined and took the whole function to the round-trip; it now delegates the
# statement (W1-loc, esbmc/esbmc#4715). Found only by censusing the *full* 5305
# Python corpus -- the stride-9 sample of docs §25 missed all five tests that
# reach it, as did every probe.
a: int = nondet_int()
__ESBMC_assume(a == 3)
print((a + 1) * 2)
assert a == 3
