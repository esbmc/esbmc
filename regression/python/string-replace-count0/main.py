def main() -> None:
    s = "aa-bb-aa".replace("aa", "x", 0)
    assert s == "aa-bb-aa"


main()
