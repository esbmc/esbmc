def main() -> None:
    parts = "abc".split("-", 2)
    assert len(parts) == 1
    assert parts[0] == "abc"


main()
