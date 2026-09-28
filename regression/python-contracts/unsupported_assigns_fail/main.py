g: int = 0

def touch(k: int) -> int:
    __ESBMC_requires(k > 0)
    __ESBMC_assigns(g)
    __ESBMC_ensures(__ESBMC_return_value > 0)
    return k

def main() -> None:
    v: int = touch(3)
    assert v == 3

main()
