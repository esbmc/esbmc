def main() -> None:
    s = "a b c"
    parts = s.split(None, -1)
    assert parts[2] == "d"

main()
