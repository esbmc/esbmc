def main() -> None:
    # 1 is not present in the slice a[1:] (only the leading element is a 1), so
    # list.index(1, 1) raises ValueError. This pins the not-found branch of the
    # new range search (modelled as a property violation).
    a = [1, 2, 3]
    assert a.index(1, 1) == 0


main()
