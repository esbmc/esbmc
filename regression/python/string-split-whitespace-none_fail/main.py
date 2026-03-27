def main() -> None:
    s = "a  b"
    parts = s.split(None)
    assert parts[1] == "a"

main()
