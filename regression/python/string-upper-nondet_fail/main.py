def main() -> None:
    s = nondet_str()
    u = s.upper()

    __ESBMC_assume(len(s) > 0)
    assert u == ""


main()
