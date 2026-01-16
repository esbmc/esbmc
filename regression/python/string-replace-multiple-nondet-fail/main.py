
def main() -> None:
    text = nondet_str()
    # Assuming text is "hello hello"
    __ESBMC_assume(len(text) == 11)
    __ESBMC_assume(text == "hello hello")

    # Test replace - intentional failure
    result = text.replace("hello", "hi")
    assert result == "hello hello"  # Wrong! Should be "hi hi"

main()
