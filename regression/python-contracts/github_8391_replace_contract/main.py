def pick(o) -> int:
    __ESBMC_requires(o > 0)
    __ESBMC_ensures(__ESBMC_return_value == 1)
    return 1

def main() -> None:
    assert True

main()
