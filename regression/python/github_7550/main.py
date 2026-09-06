# A bytes literal passed directly as an argument was rebuilt as a
# NUL-terminated char array and reinterpret-cast to the parameter's element
# pointer, so the callee read 8-byte elements out of a 3-byte object (#7550).
def first(b: bytes) -> int:
    return b[0]


def size(b: bytes) -> int:
    return len(b)


def main() -> None:
    assert first(b"\x01\x02") == 1
    assert size(b"\x01\x02") == 2

    a = b"\x01\x02"
    assert first(a) == 1
    assert first(bytes([1, 2])) == 1

    # A plain string argument must still decay as a string.
    assert len("ab") == 2


main()
