def main() -> None:
    # int.from_bytes accepts byteorder as a keyword (CPython's parameter name),
    # even though the operational model names the parameter big_endian. A bytes
    # variable receiver is used throughout (a bytes literal passed directly as
    # the argument is a separate, deferred lowering issue).
    b = bytes([1, 2])
    assert int.from_bytes(b, byteorder="big") == 258
    assert int.from_bytes(b, byteorder="little") == 513

    # The keyword form composes with the keyword-only signed argument.
    neg = bytes([255, 0])
    assert int.from_bytes(neg, byteorder="big", signed=True) == -256
    assert int.from_bytes(neg, byteorder="big", signed=False) == 65280

    # The positional form is unchanged.
    assert int.from_bytes(b, "little") == 513

    # A single byte is endianness-independent.
    one = bytes([7])
    assert int.from_bytes(one, byteorder="little") == 7


main()
