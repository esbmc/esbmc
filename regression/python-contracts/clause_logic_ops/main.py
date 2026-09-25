def sign(x: int) -> int:
    __ESBMC_requires(x > 0 and x < 100)
    __ESBMC_ensures(not (__ESBMC_return_value < 0) or x < 0)
    return x

def main() -> None:
    y: int = sign(5)
    assert y == 5

main()
