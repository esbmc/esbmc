def f(data: bytes) -> int:
    return len(data) + data[0]


x = nondet_bytes(5)
__ESBMC_assume(x[0] == 3)
assert f(x) == 9
