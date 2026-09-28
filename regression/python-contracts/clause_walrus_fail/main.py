def w(n: int) -> int:
    __ESBMC_requires((m := n) > 0)
    __ESBMC_ensures(__ESBMC_return_value == n)
    return n

def main() -> None:
    v: int = w(5)
    assert v == 5

main()
