def main() -> None:
    s = "aa-bb-aa".replace("aa", "x")
    assert s == "x-bb-x"


main()
