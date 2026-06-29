def main() -> None:
    # bytes.index / bytes.rindex are like find/rfind but raise ValueError when
    # the subsequence is absent. Both were routed through the str strncmp/strlen
    # machinery, wrong for the int-array bytes representation. Fold the literal
    # case directly: search the receiver's byte vector.
    assert bytes([1, 2, 3]).index(bytes([2, 3])) == 1
    assert bytes([1, 2, 3, 2, 3]).index(bytes([2, 3])) == 1

    # The argument may be a single integer byte (CPython accepts both forms).
    assert bytes([1, 2, 3]).index(2) == 1

    # rindex returns the last occurrence.
    assert bytes([1, 2, 1, 2]).rindex(bytes([1, 2])) == 2
    assert bytes([1, 2, 1]).rindex(1) == 2

    # The result is an int and composes in arithmetic.
    assert bytes([1, 2, 3]).index(bytes([3])) + 1 == 3

    # NUL bytes are ordinary data (strlen would have truncated at them).
    assert bytes([0, 1, 0, 2]).index(bytes([0, 2])) == 2

    # An absent subsequence raises a catchable ValueError (unlike find → -1).
    try:
        bytes([1, 2, 3]).index(bytes([9]))
        assert False
    except ValueError:
        pass
    try:
        bytes([1, 2, 3]).rindex(bytes([9]))
        assert False
    except ValueError:
        pass

    # str.index is unaffected (the char-array path).
    assert "abcabc".index("b") == 1


main()
