def main() -> None:
    s = "a,b"
    parts = s.split(",", 5)
    assert parts[1] == "c"


main()
