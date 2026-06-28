def main() -> None:
    # bytes.find / bytes.rfind over literal bytes([...]) were routed through the
    # str strncmp/strlen machinery, wrong for the int-array bytes representation.
    # Fold the literal case directly: search the receiver's byte vector.
    assert bytes([1, 2, 3]).find(bytes([2, 3])) == 1
    assert bytes([1, 2, 3]).find(bytes([9])) == -1
    assert bytes([1, 2, 3, 2, 3]).find(bytes([2, 3])) == 1

    # The argument may be a single integer byte (CPython accepts both forms).
    assert bytes([1, 2, 3]).find(2) == 1
    assert bytes([1, 2, 3]).find(9) == -1

    # An empty subsequence: find at 0, rfind at len.
    assert bytes([1, 2, 3]).find(bytes([])) == 0
    assert bytes([1, 2, 3]).rfind(bytes([])) == 3

    # rfind returns the last occurrence.
    assert bytes([1, 2, 1, 2]).rfind(bytes([1, 2])) == 2
    assert bytes([1, 2, 1]).rfind(1) == 2

    # The result is an int and composes in arithmetic.
    assert bytes([1, 2, 3]).find(bytes([3])) + 1 == 3

    # NUL bytes are ordinary data (strlen would have truncated at them).
    assert bytes([0, 1, 0, 2]).find(bytes([0, 2])) == 2

    # str.find / str.rfind are unaffected (the char-array path).
    assert "abcabc".find("b") == 1
    assert "abcabc".rfind("b") == 4


main()
