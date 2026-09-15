# chr() folds a code point into its UTF-8 bytes, so ord() must decode the whole
# sequence rather than read the first byte as a signed char (#7552).
def main() -> None:
    assert ord(chr(0)) == 0
    assert ord(chr(65)) == 65
    assert ord(chr(127)) == 127
    assert ord(chr(128)) == 128
    assert ord(chr(200)) == 200
    assert ord(chr(1114111)) == 1114111


main()
