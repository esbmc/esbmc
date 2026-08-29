def g(x: int) -> None:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > 0)
    return None

def main() -> None:
    g(1)

main()
