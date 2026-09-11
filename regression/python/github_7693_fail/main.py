def main() -> None:
    a = [(1, "abc"), (2, "def")]
    # Differs from a[0] only in the last byte: the element read has to carry
    # the whole string buffer, not just its first character.
    assert a[0] == (1, "abd")


main()
