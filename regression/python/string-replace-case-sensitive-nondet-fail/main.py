def main() -> None:
    text = nondet_str()
    # Assuming text is "Hello hello"
    __ESBMC_assume(len(text) == 11)
    __ESBMC_assume(text == "Hello hello")

    # Test replace case-sensitivity - intentional failure
    result = text.replace("hello", "hi")
    assert result == "Hi hi"  # Wrong! Should be "Hello hi"


main()
