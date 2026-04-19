def main() -> None:
    s = "a b c d"
    parts = s.split(None, 2)
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] == "c d"

main()
