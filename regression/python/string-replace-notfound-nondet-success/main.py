from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "hello"
    __ESBMC_assume(len(text) == 5)
    __ESBMC_assume(text == "hello")

    # Test replace with non-existent substring
    result = text.replace("xyz", "abc")
    assert result == "hello"  # Should remain unchanged
    assert len(result) == 5

main()
