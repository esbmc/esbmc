def main() -> None:
    s = nondet_str()
    __ESBMC_assume(len(s) > 1)
    __ESBMC_assume(s[0] == "a")

    assert s.index("a") == 1


main()
