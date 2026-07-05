def main() -> None:
    # "abcabc"[1:] == "bcabc", which does not start with "ab".
    assert "abcabc".startswith("ab", 1)


main()
