def main() -> None:
    s = "a."
    s += "b.c"
    parts = s.split(".")
    assert len(parts) == 3


main()
