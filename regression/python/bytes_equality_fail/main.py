def main() -> None:
    # Two equal-content bytes are equal; asserting they are unequal must fail.
    a = bytes([2, 3])
    b = bytes([2, 3])
    assert a != b


main()
