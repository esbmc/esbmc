def scale(x: float) -> float:
    __ESBMC_requires(x > 0.0)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(x) * 2.0)
    x = x * 2.0
    return x

def main() -> None:
    y: float = scale(1.5)
    assert y == 3.0

main()
