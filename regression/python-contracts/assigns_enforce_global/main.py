g: int = 0
h: int = 0

def touch(n: int) -> None:
    global g
    __ESBMC_requires(n > 0)
    __ESBMC_assigns(g)
    __ESBMC_ensures(g == n)
    g = n

def main() -> None:
    touch(3)

main()
