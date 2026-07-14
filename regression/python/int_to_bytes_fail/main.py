def main() -> None:
    x = 258
    b = x.to_bytes(2, "big")
    # b == b"\x01\x02"; b[0] is 1, not 9.
    assert b[0] == 9


main()
