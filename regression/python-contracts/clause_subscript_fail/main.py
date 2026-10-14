def first(l: list) -> int:
    __ESBMC_requires(True)
    __ESBMC_ensures(__ESBMC_return_value == l[0])
    return l[0]

def main() -> None:
    a = [7, 8]
    v: int = first(a)
    assert v == 7

main()
