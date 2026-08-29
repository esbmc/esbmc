def f(l: list) -> int:
    __ESBMC_requires(True)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(l))
    return 1

def main() -> None:
    a = [1, 2]
    assert f(a) == 1

main()
