def main() -> None:
    s = "a b c"
    parts = s.split(None, -1)
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] == "c"


main()
