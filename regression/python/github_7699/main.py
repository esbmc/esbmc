def main() -> None:
    # The lists differ in the tenth byte of the second element. Taking the
    # comparison length from the first element compared eight bytes and
    # reported them equal.
    a = [1, "abcdefghij"]
    b = [1, "abcdefghix"]
    assert a != b


main()
