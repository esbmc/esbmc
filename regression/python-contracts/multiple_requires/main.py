def clamp(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_requires(x < 3)
    __ESBMC_ensures(__ESBMC_return_value < 3)
    return x

def main() -> None:
    y: int = clamp(2)
    assert y == 2

main()
