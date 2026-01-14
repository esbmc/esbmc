from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "hello"
    __ESBMC_assume(len(text) == 5)
    __ESBMC_assume(text == "hello")

    # Test replace making string shorter
    result = text.replace("hello", "hi")
    assert result == "hi"
    assert len(result) == 2

main()
