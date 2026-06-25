def main() -> None:
    # The 2-argument form defaults signed=False (CPython), so it no longer
    # raises a missing-argument TypeError.
    b = b"\x01\x02"
    assert int.from_bytes(b, "big") == 258
    assert int.from_bytes(b, "little") == 513

    # signed=False keyword matches the default.
    assert int.from_bytes(b, "big", signed=False) == 258

    # A leading zero byte and the single-byte case.
    c = b"\x00\xff"
    assert int.from_bytes(c, "big") == 255
    d = b"\x2a"
    assert int.from_bytes(d, "big") == 42


main()
