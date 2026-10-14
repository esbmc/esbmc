def main() -> None:
    # Instance form on an int variable, big-endian.
    x = 258
    b = x.to_bytes(2, "big")
    assert b[0] == 1 and b[1] == 2 and len(b) == 2

    # Little-endian reverses the byte order.
    c = x.to_bytes(2, "little")
    assert c[0] == 2 and c[1] == 1

    # A wider value, little-endian (1000 == 0x03E8).
    d = (1000).to_bytes(2, "little")
    assert d[0] == 232 and d[1] == 3

    # The int.to_bytes(value, ...) class form.
    e = int.to_bytes(258, 2, "big")
    assert e[0] == 1 and e[1] == 2

    # Zero padding to the requested length.
    f = (5).to_bytes(4, "big")
    assert f[0] == 0 and f[1] == 0 and f[2] == 0 and f[3] == 5


main()
