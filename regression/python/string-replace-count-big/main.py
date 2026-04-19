def main() -> None:
    s = "aa-bb-aa".replace("aa", "x", 5)
    assert s == "x-bb-x"


main()
