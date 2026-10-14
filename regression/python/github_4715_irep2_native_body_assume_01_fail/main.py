def main() -> None:
    x: int = 5
    __ESBMC_assume(x > 0)
    assert x > 10


main()
