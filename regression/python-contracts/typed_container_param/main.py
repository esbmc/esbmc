def take(l: list, n: int) -> int:
    __ESBMC_requires(n > 0)
    __ESBMC_ensures(__ESBMC_return_value == n)
    return n

def main() -> None:
    a = [7, 8, 9]
    v: int = take(a, 3)
    assert v == 3

main()
