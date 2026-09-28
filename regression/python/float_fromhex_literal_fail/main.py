def main() -> None:
    # float.fromhex("0x1.8p3") is 12.0; asserting it equals a different value
    # is provably FAILED now that fromhex folds exactly.
    assert float.fromhex("0x1.8p3") == 3.5


main()
