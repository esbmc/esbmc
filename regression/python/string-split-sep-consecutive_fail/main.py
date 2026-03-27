def main() -> None:
    s = "a,,b"
    parts = s.split(",")
    assert parts[1] == "x"


main()
