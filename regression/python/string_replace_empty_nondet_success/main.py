from esbmc import nondet_str

def main() -> None:
    text = nondet_str()
    # Assuming text is "hello"
    __ESBMC_assume(len(text) == 5)
    __ESBMC_assume(text == "hello")

    # Test replace with empty old string (no replacement should occur)
    result = text.replace("", "x")
    # Empty old string in Python inserts new_str between each char
    # "hello".replace("", "x") -> "xhxexlxlxox"
    assert len(result) == 11

    # Test replace with empty new string (removes old string)
    result2 = text.replace("l", "")
    assert result2 == "heo"

main()
