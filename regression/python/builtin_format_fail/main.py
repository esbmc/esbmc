def main() -> None:
    # format(255, "x") == "ff", not "00".
    assert format(255, "x") == "00"


main()
