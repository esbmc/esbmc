def h(x: int) -> int:
    __ESBMC_requires(x > 0, x < 10)
    return x

def main() -> None:
    y: int = h(5)
    assert y == 5

main()
