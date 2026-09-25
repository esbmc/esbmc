class Counter:
    def __init__(self) -> None:
        self.n: int = 0

    def add(self, k: int) -> int:
        __ESBMC_requires(k > 0)
        __ESBMC_ensures(__ESBMC_return_value < 0)
        return k

def main() -> None:
    c = Counter()
    v: int = c.add(3)
    assert v == 3

main()
