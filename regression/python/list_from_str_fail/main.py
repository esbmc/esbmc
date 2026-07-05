def main() -> None:
    x = list("abc")
    # x == ['a', 'b', 'c']; x[0] is "a", not "z".
    assert x[0] == "z"


main()
