def check(c: int) -> None:
    assert c >= 0

a: int = nondet_int()
__ESBMC_assume(a >= -3)
check(a // 2)
