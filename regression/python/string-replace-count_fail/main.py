def main() -> None:
    s = "aa-bb-aa".replace("aa", "x", 1)
    assert s == "x-bb-x"


main()
