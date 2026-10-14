def main() -> None:
    # rpartition splits at the LAST '.', so [0] is "a.b", not "a".
    p = "a.b.c".rpartition(".")
    assert p[0] == "a"


main()
