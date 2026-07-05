def main() -> None:
    # [1, 2, 3].insert(-1, 9) gives [1, 2, 9, 3], not [1, 2, 3, 9].
    a = [1, 2, 3]
    a.insert(-1, 9)
    assert a == [1, 2, 3, 9]


main()
