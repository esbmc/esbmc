def main() -> None:
    s = "a,b,c"
    parts = s.split(",", 1)
    assert parts[1] == "c"

main()
