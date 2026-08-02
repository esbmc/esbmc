def main() -> None:
    # Without maxsplit, rsplit == split.
    assert "a.b.c".rsplit(".") == ["a", "b", "c"]

    # maxsplit counts separators from the RIGHT.
    assert "a.b.c".rsplit(".", 1) == ["a.b", "c"]

    # maxsplit == 0 yields the whole string.
    assert "a.b.c".rsplit(".", 0) == ["a.b.c"]

    # Consecutive separators produce an empty piece (right-anchored).
    assert "a..b".rsplit(".", 1) == ["a.", "b"]

    # A typed-variable receiver infers the list result type correctly.
    s: str = "x-y-z"
    assert s.rsplit("-", 1) == ["x-y", "z"]


main()
