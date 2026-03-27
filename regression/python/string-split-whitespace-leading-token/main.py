def main() -> None:
    s = "     token"
    parts = s.split(None, 1)
    assert parts[0] == "token"

main()
