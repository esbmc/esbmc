def main() -> None:
    s = "a"
    parts = s.split(",")
    assert len(parts) == 1
    assert parts[0] == "a"


main()
