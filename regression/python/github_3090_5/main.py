i1: int = nondet_int()
i2: int = nondet_int()
i3: int = nondet_int()
__ESBMC_assume(i1 == 102)
__ESBMC_assume(i2 == 111)
__ESBMC_assume(i3 == 111)
s: str = ""
s += chr(i1)
s += chr(i2)
s += chr(i3)
assert s == "foo"
