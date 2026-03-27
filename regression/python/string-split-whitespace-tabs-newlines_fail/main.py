def main() -> None:
    s = "\t\na\t b\n\tc"
    parts = s.split(None, 1)
    assert parts[1] == "z"

main()
