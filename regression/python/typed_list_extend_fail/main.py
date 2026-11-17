def main() -> None:
    n = nondet_int()
    if n < 0 or n > 3:
        return
    a = [7]
    a.extend([0] * n)
    assert a[n] == 7


main()
