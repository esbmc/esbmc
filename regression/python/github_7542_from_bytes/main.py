# A range loop visits its body twice; the second visit re-folded the already
# folded byteorder and turned "big" into little-endian (#7542). Nested range
# loops visit the inner body more than twice.
def main() -> None:
    a = bytes([1, 0])
    for _ in range(1):
        assert int.from_bytes(a, "big") == 256
        assert int.from_bytes(a, "little") == 1
        for _ in range(1):
            assert int.from_bytes(a, "big") == 256


main()
