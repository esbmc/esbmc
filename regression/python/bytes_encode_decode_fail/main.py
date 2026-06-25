def main() -> None:
    # b"ab".decode() == "ab", not "xy".
    assert b"ab".decode() == "xy"


main()
