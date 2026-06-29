def main() -> None:
    # bytes([1, 2, 3]).index(bytes([2, 3])) == 1, not 9.
    assert bytes([1, 2, 3]).index(bytes([2, 3])) == 9


main()
