def main() -> None:
    # bytes are modelled as wide-int arrays, and a bytes slice previously kept a
    # phantom null terminator and routed len() through strlen, which stopped at
    # the first element's zero high bytes — so len(b[1:3]) returned 1, not 2.
    b = bytes([1, 2, 3, 4])

    # Inline and variable forms both report the correct element count.
    assert len(b[1:3]) == 2
    s = b[1:3]
    assert len(s) == 2 and s[0] == 2 and s[1] == 3

    # Open-ended slices.
    assert len(b[:2]) == 2
    assert len(b[2:]) == 2

    # Embedded zero bytes are counted (strlen would have stopped early).
    z = bytes([5, 0, 7, 0, 9])
    w = z[1:4]
    assert len(w) == 3 and w[0] == 0 and w[1] == 7 and w[2] == 0


main()
