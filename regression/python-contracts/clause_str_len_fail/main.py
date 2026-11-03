def f(s: str) -> int:
    __ESBMC_requires(len(s) > 0)
    return 1

def main() -> None:
    v: int = f("ab")
    assert v == 1

main()
