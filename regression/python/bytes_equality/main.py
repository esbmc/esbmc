def main() -> None:
    # bytes are value sequences: == / != compare content, not identity. Two
    # equal-content bytes stored in separate variables previously compared by
    # base address (&a[0] == &b[0]) and so were wrongly unequal.
    a = bytes([2, 3])
    b = bytes([2, 3])
    assert a == b
    assert not (a != b)
    assert a == bytes([2, 3])

    # Different content / length compare unequal.
    assert bytes([2, 3]) != bytes([2, 4])
    assert not (bytes([2, 3]) == bytes([2, 4]))
    assert bytes([2, 3]) != bytes([2, 3, 4])

    # Empty bytes are equal; byte-string literals behave the same.
    assert bytes([]) == bytes([])
    c = b"AB"
    d = b"AB"
    assert c == d


main()
