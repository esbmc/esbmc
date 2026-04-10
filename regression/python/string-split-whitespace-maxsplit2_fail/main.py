def main() -> None:
    s = "a b c d"
    parts = s.split(None, 2)
    assert parts[2] == "d"

main()
