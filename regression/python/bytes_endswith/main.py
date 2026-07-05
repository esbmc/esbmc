def main() -> None:
    # bytes.endswith over a literal bytes([...]) was routed through the str
    # strncmp/strlen machinery, which is wrong for the int-array bytes
    # representation, so it silently mis-computed its from-the-end offset.
    # The fold compares the byte vectors of the literal operands directly.
    assert bytes([1, 2, 3]).endswith(bytes([2, 3])) is True
    assert bytes([1, 2, 3]).endswith(bytes([1, 2])) is False
    assert bytes([1, 2, 3]).endswith(bytes([1, 2, 3])) is True
    assert bytes([1, 2, 3]).endswith(bytes([3])) is True

    # A suffix longer than the receiver does not match.
    assert bytes([1, 2]).endswith(bytes([0, 1, 2])) is False

    # NUL bytes are ordinary data here (strlen would have truncated at them).
    assert bytes([0, 1, 0]).endswith(bytes([1, 0])) is True
    assert bytes([1, 0, 2]).endswith(bytes([0, 2])) is True

    # startswith over literals is folded the same way (it was already correct
    # via strncmp; the fold keeps it correct and consistent).
    assert bytes([1, 2, 3]).startswith(bytes([1, 2])) is True
    assert bytes([1, 2, 3]).startswith(bytes([2, 3])) is False

    # str.endswith is unaffected (the char-array path).
    assert "hello".endswith("lo") is True
    assert "hello".endswith("he") is False


main()
