def main() -> None:
    parts = "a-b-c".split("-", 1)
    assert len(parts) == 2
    assert parts[0][0] == "a"
    assert parts[1][0] == "b"
    assert parts[1][1] == "-"
    assert parts[1][2] == "c"
    assert parts[0][1] == "\0"
    assert parts[1][3] == "\0"


main()
