def main() -> None:
    # (42).__str__() is "42", not "43".
    x = 42
    assert x.__str__() == "43"


main()
