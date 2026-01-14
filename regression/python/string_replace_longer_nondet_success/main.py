from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "hi"
    __ESBMC_assume(len(text) == 2)
    __ESBMC_assume(text == "hi")

    # Test replace making string longer
    result = text.replace("hi", "hello")
    assert result == "hello"
    assert len(result) == 5

main()
