def last(l: list) -> int:
    __ESBMC_requires(len(l) > 0)
    __ESBMC_ensures(__ESBMC_return_value == l[-1])
    return l[0]

def main() -> None:
    a = [7, 8, 9]
    v: int = last(a)
    assert v == 7

main()
