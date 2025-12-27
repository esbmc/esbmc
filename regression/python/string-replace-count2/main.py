def main() -> None:
    s = "aa-aa-aa".replace("aa", "x", 2)
    assert s == "x-x-aa"


main()
