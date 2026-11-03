def g() -> int:
    return 42


def h() -> int:
    return 7


def call(k: int) -> int:
    return ls[k]()


ls = [g, h]
k = nondet_int()
__ESBMC_assume(k >= 0 and k <= 1)
a = call(k)
assert a == 42
