from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "Hello HELLO hello"
    __ESBMC_assume(len(text) == 17)
    __ESBMC_assume(text == "Hello HELLO hello")

    # Test replace is case-sensitive
    result = text.replace("hello", "hi")
    assert result == "Hello HELLO hi"  # Only lowercase "hello" replaced

main()
