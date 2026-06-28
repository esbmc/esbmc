def main() -> None:
    # bytes.fromhex("0102")[0] == 1, not 9.
    b = bytes.fromhex("0102")
    assert b[0] == 9


main()
