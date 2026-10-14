# len() of chr(200) is one code point, not its two UTF-8 bytes (#7552).
def main() -> None:
    assert len(chr(200)) == 2


main()
