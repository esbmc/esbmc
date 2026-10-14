g: int = 0
h: int = 0

def touch(n: int) -> None:
    __ESBMC_requires(n > 0)
    __ESBMC_assigns(g)
    __ESBMC_ensures(True)
    global g
    g = n

def main() -> None:
    global g, h
    g = 5
    h = 5
    touch(3)
    assert g == 5

main()
