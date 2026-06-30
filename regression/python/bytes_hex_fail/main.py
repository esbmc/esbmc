def main() -> None:
    # b"\x01\x02".hex() == "0102", not "0103".
    assert b"\x01\x02".hex() == "0103"


main()
