def size(l: list) -> int:
    __ESBMC_requires(len(l) > 0)
    __ESBMC_ensures(__ESBMC_return_value == len(l))
    return len(l) + 1

def main() -> None:
    a = [7, 8, 9]
    v: int = size(a)
    assert v > 0

main()
