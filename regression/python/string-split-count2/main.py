def main() -> None:
    parts = "a-b-c".split("-", 0)
    assert len(parts) == 1
    assert parts[0][0] == "a"
    assert parts[0][1] == "-"
    assert parts[0][2] == "b"
    assert parts[0][3] == "-"
    assert parts[0][4] == "c"
    assert parts[0][5] == "\0"


main()
