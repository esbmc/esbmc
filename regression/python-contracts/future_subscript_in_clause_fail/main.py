def first(l: list) -> int:
    __ESBMC_ensures(__ESBMC_return_value == l[0])
    return l[1]

def main() -> None:
    a = [7, 8, 9]
    v: int = first(a)
    assert v == 7

main()
