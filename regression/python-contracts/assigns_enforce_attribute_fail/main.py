class Counter:
    def __init__(self) -> None:
        self.n: int = 0
        self.m: int = 0

def bump(c: Counter) -> None:
    __ESBMC_requires(c.n >= 0)
    __ESBMC_assigns(c.n)
    __ESBMC_ensures(True)
    c.n = c.n + 1
    c.m = 3

def main() -> None:
    c = Counter()
    bump(c)

main()
