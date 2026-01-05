def main() -> None:
    parts = "a-b-c".split("-", -1)
    assert len(parts) == 3
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] == "c"


main()
