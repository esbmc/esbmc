def main() -> None:
    n = nondet_int()
    if n > 4:
        return
    a = n * [0]
    assert len(a) == n or n > 0


main()
