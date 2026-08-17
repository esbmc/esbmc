def flip(b: bool) -> bool:
    __ESBMC_requires(b)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(b))
    b = not b
    return b

def main() -> None:
    y: bool = flip(True)
    assert y == False

main()
