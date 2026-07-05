def main() -> None:
    # int.to_bytes accepts length and byteorder positionally or by keyword, and
    # both default (length=1, byteorder='big' since CPython 3.11). The handler
    # previously required them positionally and rejected the keyword/default forms.
    x = 258

    # byteorder as a keyword (length still positional).
    bk = x.to_bytes(2, byteorder="big")
    assert bk[0] == 1 and bk[1] == 2

    # both length and byteorder as keywords.
    bb = x.to_bytes(length=2, byteorder="big")
    assert bb[0] == 1 and bb[1] == 2

    # little-endian keyword.
    bl = x.to_bytes(2, byteorder="little")
    assert bl[0] == 2 and bl[1] == 1

    # length only (byteorder defaults to 'big').
    bp = x.to_bytes(2)
    assert bp[0] == 1 and bp[1] == 2

    # no arguments (length=1, byteorder='big').
    bd = (5).to_bytes()
    assert bd[0] == 5 and len(bd) == 1

    # the unbound type-method form with keywords.
    bt = int.to_bytes(258, length=2, byteorder="big")
    assert bt[0] == 1 and bt[1] == 2

    # the positional form is unchanged.
    bpos = x.to_bytes(2, "big")
    assert bpos[0] == 1 and bpos[1] == 2


main()
