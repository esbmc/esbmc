def foo() -> None:
    a = int(2)
    assert a == 2


foo()

x = int(1)
assert x == 1

z = int(1.1111)
assert z == 1

z = int(2.1111) * 2
assert z == 4

a: int = __VERIFIER_nondet_int()
__ESBMC_assume(a > 0 and a < 10)
z = int(1.1111) * a
assert z > 0 and z < 10

y = bool(True)
assert y == True
