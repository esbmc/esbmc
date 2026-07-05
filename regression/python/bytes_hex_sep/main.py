def main() -> None:
    # bytes.hex(sep) inserts a one-character separator between every byte.
    assert bytes([1, 2, 3]).hex("-") == "01-02-03"
    assert b"\x01\xab".hex(" ") == "01 ab"

    # bytes_per_sep groups bytes: a positive count groups from the right,
    # a negative count from the left (CPython semantics).
    c = bytes([0xb9, 0x01, 0x9e, 0xf3])
    assert c.hex("_", 2) == "b901_9ef3"
    d = bytes([1, 2, 3, 4, 5])
    assert d.hex("_", 2) == "01_0203_0405"
    assert d.hex(":", -2) == "0102:0304:05"

    # bytes_per_sep == 0 inserts no separator; a single byte has none either.
    assert bytes([1, 2]).hex("-", 0) == "0102"
    assert bytes([0xab]).hex("-") == "ab"

    # The no-argument form is unchanged, and the result composes as a str.
    assert c.hex() == "b9019ef3"
    r = c.hex("_", 2)
    assert len(r) == 9
    assert r[0] == "b" and r[4] == "_"


main()
