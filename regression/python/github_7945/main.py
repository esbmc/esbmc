# The byteorder string is folded at the resolved callee, so an aliased call
# is covered (#7945).
def main() -> None:
    f = int.from_bytes
    a = bytes([1, 0])
    assert f(a, "little") == 1
    assert f(a, byteorder="little") == 1
    assert f(a) == 256
    for _ in range(1):
        assert int.from_bytes(a, "big") == 256
        assert int.from_bytes(a, "little") == 1


main()
