def main() -> None:
    # 1 is not present in the slice a[1:] (only the leading element is a 1), so
    # list.index(1, 1) raises ValueError. This pins the not-found branch of the
    # range search; the uncaught ValueError makes verification fail here (the
    # assert would otherwise trip on the a.index(1, 1) == 0 mismatch anyway).
    a = [1, 2, 3]
    assert a.index(1, 1) == 0


main()
