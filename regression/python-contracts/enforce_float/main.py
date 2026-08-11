def half(x: float) -> float:
    __ESBMC_requires(x > 0.0)
    __ESBMC_ensures(__ESBMC_return_value < x)
    return x / 2.0

def main() -> None:
    z: float = half(4.0)
    assert z < 4.0

main()
