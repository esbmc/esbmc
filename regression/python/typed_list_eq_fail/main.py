def main() -> None:
    n = nondet_int()
    x = nondet_int()
    if n < 1 or n > 3:
        return
    a = [x] * n
    b = [0] * n
    assert a == b


main()
