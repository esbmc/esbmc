# chr(n) above 127 is still stored as its multi-byte UTF-8 form, so len() counts
# encoding units instead of characters. Fixing that needs a code-point string
# representation, not a change to ord() (#7552).
def main() -> None:
    assert len(chr(200)) == 1


main()
