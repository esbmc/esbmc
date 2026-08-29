def double(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 0 - x

def main() -> None:
    y: int = double(5)
    assert y == -5

main()
