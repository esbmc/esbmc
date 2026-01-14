from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "aaa"
    __ESBMC_assume(len(text) == 3)
    __ESBMC_assume(text == "aaa")

    # Test replace with count on nondet_string
    result = text.replace("a", "b", 2)
    assert result == "bba"

    # Test replace all
    result2 = text.replace("a", "c")
    assert result2 == "ccc"

main()
