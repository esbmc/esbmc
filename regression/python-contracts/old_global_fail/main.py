g: int = 0

def touch(k: int) -> int:
    global g
    __ESBMC_requires(k > 0)
    __ESBMC_ensures(g == __ESBMC_old(g))
    g = g + 1
    return k

def main() -> None:
    v: int = touch(3)
    assert v == 3

main()
