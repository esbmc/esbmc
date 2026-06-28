def main() -> None:
    # int.from_bytes(b, byteorder="big") == 258 for b == bytes([1, 2]), not 257.
    b = bytes([1, 2])
    assert int.from_bytes(b, byteorder="big") == 257


main()
