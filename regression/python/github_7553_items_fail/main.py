# An items view holds (key, value) pairs, so it never equals a set of bare
# keys. It lowers to the same `keys` member as a keys view, so a comparison
# keyed on the member name alone would wrongly prove this one (#7553).
def main() -> None:
    assert {1: 1}.items() == {1}


main()
