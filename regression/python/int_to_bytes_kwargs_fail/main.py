def main() -> None:
    # 258 big-endian is b'\x01\x02', so b[0] == 1, not 2.
    b = (258).to_bytes(length=2, byteorder="big")
    assert b[0] == 2


main()
