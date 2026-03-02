def main() -> None:
    x: int = nondet_int()
    if x > 0:
        assert x < 0


main()
