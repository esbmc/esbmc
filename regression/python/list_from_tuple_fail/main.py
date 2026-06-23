def main() -> None:
    a = list((2, 3))
    # a == [2, 3]; a[0] is 2, not 9.
    assert a[0] == 9


main()
