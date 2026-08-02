def main() -> None:
    # "Hello world" is not titlecased ('w' follows an uncased space but is
    # lowercase), so this assertion must be falsified (it was unsupported ->
    # spurious FAILED before istitle was modelled, and is a real FAILED now).
    assert "Hello world".istitle()


main()
