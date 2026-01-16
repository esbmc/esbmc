
def main() -> None:
    text = nondet_str()
    # Keep the symbolic domain small to avoid long solver runs.
    __ESBMC_assume(len(text) == 1)
    __ESBMC_assume(text == "a")

    # Test replace with empty old string (no replacement should occur)
    result = text.replace("", "x")
    # Empty old string in Python inserts new_str between each char
    # "a".replace("", "x") -> "xax"
    assert len(result) == 3

main()
