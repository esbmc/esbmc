def main() -> None:
    # bytes.hex() returns the lowercase hex string of the byte values.
    assert b"\x01\x02\xab".hex() == "0102ab"

    # A bytes value held in a variable works too, and the result is a str.
    b = b"\xde\xad\xbe\xef"
    s = b.hex()
    assert s == "deadbeef"
    assert len(s) == 8
    assert s[0] == "d" and s[7] == "f"

    # ASCII bytes and the empty case.
    assert b"AB".hex() == "4142"
    assert b"".hex() == ""

    # The result composes as a normal string.
    assert b"\x0f".hex() + "!" == "0f!"


main()
