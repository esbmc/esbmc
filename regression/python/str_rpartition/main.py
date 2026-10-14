def main() -> None:
    # rpartition splits at the LAST occurrence of the separator.
    p = "a.b.c".rpartition(".")
    assert p[0] == "a.b"
    assert p[1] == "."
    assert p[2] == "c"
    assert len(p) == 3

    # When the separator is absent, rpartition returns ("", "", original)
    # (the unmatched receiver goes in the last element, unlike partition).
    q = "abc".rpartition(".")
    assert q[0] == ""
    assert q[1] == ""
    assert q[2] == "abc"

    # Multi-character separator: split at the last "xx".
    r = "xxabxxcd".rpartition("xx")
    assert r[0] == "xxab"
    assert r[1] == "xx"
    assert r[2] == "cd"

    # Empty receiver: nothing to split.
    e = "".rpartition(".")
    assert e[0] == "" and e[1] == "" and e[2] == ""


main()
