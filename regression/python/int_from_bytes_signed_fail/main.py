def main() -> None:
    a = bytes([255, 255])
    # Two's-complement 0xFFFF signed is -1, not 65535.
    assert int.from_bytes(a, "big", signed=True) == 65535


main()
