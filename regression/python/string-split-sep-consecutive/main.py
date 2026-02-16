def main() -> None:
    s = "a,,b"
    parts = s.split(",")
    assert parts[0] == "a"
    assert parts[1] == ""
    assert parts[2] == "b"

main()
