g: int = 0

def touch(k: int) -> int:
    __ESBMC_requires(k > 0)
    __ESBMC_ensures(g == __ESBMC_old(g))
    return k

def main() -> None:
    v: int = touch(3)
    assert v == 3

main()
