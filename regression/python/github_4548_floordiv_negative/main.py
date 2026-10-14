def check(c: int) -> None:
    assert c == -3

a: int = nondet_int()
__ESBMC_assume(a == -5)
check(a // 2)
