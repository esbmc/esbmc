def main() -> None:
    # "a\tb" contains a tab, which is not printable: isprintable() is False, so
    # this assertion must be falsified (it was unsupported -> spurious FAILED
    # before isprintable was modelled, and is a real FAILED now).
    assert "a\tb".isprintable()


main()
