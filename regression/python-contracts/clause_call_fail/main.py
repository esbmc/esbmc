def size(l: list) -> int:
    __ESBMC_requires(len(l) > 0)
    __ESBMC_ensures(__ESBMC_return_value >= 0)
    return len(l)

def main() -> None:
    a = [7, 8]
    v: int = size(a)
    assert v == 2

main()
