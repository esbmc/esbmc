def main() -> None:
    # bytes.fromhex parses a constant hex string into a bytes object.
    b = bytes.fromhex("0102ab")
    assert b[0] == 1
    assert b[1] == 2
    assert b[2] == 171
    assert len(b) == 3

    # CPython skips ASCII whitespace between byte pairs.
    spaced = bytes.fromhex("de ad be ef")
    assert spaced[0] == 222
    assert spaced[3] == 239
    assert len(spaced) == 4

    # Uppercase digits are accepted, and the result composes with .hex().
    assert bytes.fromhex("DEAD").hex() == "dead"

    # The empty string yields an empty bytes object.
    empty = bytes.fromhex("")
    assert len(empty) == 0

    # A variable holding the hex string works too.
    s = "ff00"
    v = bytes.fromhex(s)
    assert v[0] == 255 and v[1] == 0


main()
