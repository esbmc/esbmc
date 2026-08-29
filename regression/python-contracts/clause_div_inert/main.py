def safe(d: int) -> int:
    __ESBMC_requires(100 // d > 0)
    return 1

def main() -> None:
    x: int = nondet_int()
    assert safe(x) == 1

main()
