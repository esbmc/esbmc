def main() -> None:
    s = "ab".replace("", "x")
    assert s == "xabx"


main()
