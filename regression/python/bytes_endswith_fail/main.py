def main() -> None:
    # bytes([1, 2, 3]) ends with bytes([2, 3]), so endswith is True, not False.
    assert bytes([1, 2, 3]).endswith(bytes([2, 3])) is False


main()
