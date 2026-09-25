def main() -> None:
    t = nondet_str()
    __ESBMC_assume(len(t) == 3)
    assert False


main()
