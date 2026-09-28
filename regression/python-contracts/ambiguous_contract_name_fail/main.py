class Counter:
    def bump(self, k: int) -> int:
        __ESBMC_requires(k > 0)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def bump(k: int) -> int:
    __ESBMC_requires(k > 0)
    __ESBMC_ensures(__ESBMC_return_value > 0)
    return k

def main() -> None:
    c = Counter()
    a: int = c.bump(1)
    b: int = bump(2)
    assert a == 1 and b == 2

main()
