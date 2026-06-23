def main() -> None:
    t = tuple("ab")
    # t == ('a', 'b'); t[0] is "a", not "z".
    assert t[0] == "z"


main()
