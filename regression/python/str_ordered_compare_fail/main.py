def main() -> None:
    # "abc" <= "abc" is True; asserting it is False must fail. (Pre-fix this
    # wrongly verified because <= folded to `!equal` == False.)
    assert not ("abc" <= "abc")


main()
