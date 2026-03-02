def main() -> None:
    s = nondet_str()
    r = s.rstrip()

    __ESBMC_assume(len(s) > 0)
    assert len(r) > 0


main()
