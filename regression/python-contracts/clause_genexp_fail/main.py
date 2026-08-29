def allpos(l: list, n: int) -> int:
    __ESBMC_requires(all(x > 0 for x in l))
    __ESBMC_ensures(__ESBMC_return_value == n)
    return n

def main() -> None:
    a = [1, 2]
    v: int = allpos(a, 5)
    assert v == 5

main()
