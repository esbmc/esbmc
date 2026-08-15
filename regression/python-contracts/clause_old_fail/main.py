def bump(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(x) + 1)
    return x + 1

def main() -> None:
    y: int = bump(5)
    assert y == 6

main()
