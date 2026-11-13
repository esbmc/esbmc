def build(n: int) -> list[int]:
    out = []
    i = 0
    while i < n:
        out.append(i)
        i += 1
    return out


n = nondet_int()
__ESBMC_assume(0 <= n and n <= 3)
assert sum(build(n)) == n * (n - 1) // 2
