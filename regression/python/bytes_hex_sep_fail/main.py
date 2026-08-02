def main() -> None:
    # bytes([1, 2, 3]).hex("-") == "01-02-03", not "010203".
    assert bytes([1, 2, 3]).hex("-") == "010203"


main()
