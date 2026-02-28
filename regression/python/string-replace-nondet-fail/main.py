def main() -> None:
    text = nondet_str()
    # Assuming text is "hello world"
    __ESBMC_assume(len(text) == 11)
    __ESBMC_assume(text == "hello world")

    # Test replace with nondet_string - intentional failure
    result = text.replace("world", "Python")
    assert result == "hello world"  # Wrong! Should be "hello Python"


main()
