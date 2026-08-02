def main() -> None:
    # startswith/endswith with a start position: s.startswith(p, start)
    # is s[start:].startswith(p).
    assert "abcabc".startswith("bc", 1)
    assert not "abcabc".startswith("ab", 1)
    assert "hello".startswith("lo", 3)

    # With a start and end: s[start:end].
    assert "abcabc".startswith("ab", 0, 5)
    assert not "hello".startswith("lo", 3, 4)

    # Negative indices and out-of-range starts.
    assert "abcdef".startswith("cd", 2, -1)
    assert not "ab".startswith("x", 5)
    assert "abc".startswith("", 1)

    # An empty affix matches at start == len, but NOT past it (CPython).
    assert "abc".startswith("", 3)
    assert not "abc".startswith("", 4)
    assert not "abc".endswith("", 4)

    # endswith with positions: s[start:end].endswith(suffix).
    assert "abcabc".endswith("bc", 3)
    assert "file.py".endswith("le", 0, 4)


main()
