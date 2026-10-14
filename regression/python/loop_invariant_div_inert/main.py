def main() -> None:
    d: int = nondet_int()
    i: int = 0
    while i < 3:
        __ESBMC_loop_invariant(100 // d > 0)
        i = i + 1
    assert i == 3

main()
