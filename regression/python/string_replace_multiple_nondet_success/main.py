from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "hello hello"
    __ESBMC_assume(len(text) == 11)
    __ESBMC_assume(text == "hello hello")

    # Test replace all occurrences
    result = text.replace("hello", "hi")
    assert result == "hi hi"
    assert len(result) == 5

main()
