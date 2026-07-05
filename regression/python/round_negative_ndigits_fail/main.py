def main() -> None:
    # round(1234, -2) is 1200, not 1300.
    assert round(1234, -2) == 1300


main()
