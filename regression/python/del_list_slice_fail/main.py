def main() -> None:
    # del a[1:3] removes indices 1 and 2, leaving [1, 4] — not [1, 2, 4].
    a = [1, 2, 3, 4]
    del a[1:3]
    assert a == [1, 2, 4]


main()
