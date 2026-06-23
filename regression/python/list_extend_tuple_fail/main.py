def main() -> None:
    a = [1]
    a.extend((2, 3))
    # a == [1, 2, 3]; a[2] is 3, not 99.
    assert a[2] == 99


main()
