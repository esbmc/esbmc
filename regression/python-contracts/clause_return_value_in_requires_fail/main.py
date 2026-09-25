def k(x: int) -> int:
    __ESBMC_requires(__ESBMC_return_value > 0)
    return x

def main() -> None:
    y: int = k(5)
    assert y == 5

main()
