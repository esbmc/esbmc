def f(l: list) -> int:
    __ESBMC_requires(l[5] > 0)
    __ESBMC_ensures(__ESBMC_return_value == 1)
    return 1

def main() -> None:
    a = [7, 8, 9]
    v: int = f(a)
    assert v == 1

main()
