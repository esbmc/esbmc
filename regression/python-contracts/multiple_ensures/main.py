def twice(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    __ESBMC_ensures(__ESBMC_return_value == 2 * x)
    return 2 * x

def main() -> None:
    y: int = twice(4)
    assert y == 8

main()
