def f(x: int) -> int:
    __ESBMC_requires(x > 0)
    return 100 // x

def main() -> None:
    n: int = nondet_int()
    y: int = f(n)

main()
