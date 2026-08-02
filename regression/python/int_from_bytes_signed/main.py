def main() -> None:
    # signed=True interprets the most-significant byte's MSB as the sign:
    # byte 0 for big-endian, the last byte for little-endian.
    a = bytes([255, 255])
    assert int.from_bytes(a, "big", signed=True) == -1

    # The sign byte is byte 0 for big-endian (here 0xFF -> negative).
    b = bytes([255, 0])
    assert int.from_bytes(b, "big", signed=True) == -256

    # Little-endian: the last byte carries the sign.
    c = bytes([0, 255])
    assert int.from_bytes(c, "little", signed=True) == -256

    # Single byte, and the positive signed cases.
    d = bytes([128])
    assert int.from_bytes(d, "big", signed=True) == -128
    e = bytes([1, 2])
    assert int.from_bytes(e, "big", signed=True) == 258

    # signed=False is unaffected.
    assert int.from_bytes(e, "big", signed=False) == 258


main()
