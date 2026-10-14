def main() -> None:
    # divmod(17, 5) == (3, 2), not (9, 9).
    x = divmod(17, 5)
    assert x == (9, 9)


main()
