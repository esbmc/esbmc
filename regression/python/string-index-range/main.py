def main() -> None:
    s = "banana"
    assert s.index("na", 3) == 4
    assert s.index("na", 0, 4) == 2


main()
