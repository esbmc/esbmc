def second(l: list) -> float:
    __ESBMC_ensures(__ESBMC_return_value == l[1])
    return l[1]

def main() -> None:
    v: float = second([1.5, 2.5])
    assert v == 2.5

main()
