def main() -> None:
    s = "a,b"
    parts = s.split(",", 5)
    assert parts[0] == "a"
    assert parts[1] == "b"


main()
