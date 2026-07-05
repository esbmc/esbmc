def main() -> None:
    b = b"\x01\x02"
    # int.from_bytes(b, "big") == 258, not 999.
    assert int.from_bytes(b, "big") == 999


main()
