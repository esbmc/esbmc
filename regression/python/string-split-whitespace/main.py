def main() -> None:
    s = "  a   b\tc\n"
    parts = s.split()
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] == "c"


main()
