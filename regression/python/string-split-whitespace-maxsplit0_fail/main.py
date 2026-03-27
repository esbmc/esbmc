def main() -> None:
    s = "a b c"
    parts = s.split(None, 0)
    assert parts[0] == "z"


main()
