def first(l: list, i: int) -> int:
    __ESBMC_requires(True)
    __ESBMC_ensures(__ESBMC_return_value == l[i])
    return l[i]

def main() -> None:
    a = [7, 8]
    v: int = first(a, 0)
    assert v == 7

main()
