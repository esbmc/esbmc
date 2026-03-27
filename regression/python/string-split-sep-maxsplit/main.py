def main() -> None:
    s = "a,b,c"
    parts = s.split(",", 1)
    assert parts[0] == "a"
    assert parts[1] == "b,c"

main()
