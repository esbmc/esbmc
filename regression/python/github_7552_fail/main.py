# -61 is 0xC3 read as a signed byte: the first byte of chr(200)'s UTF-8 form.
# ord() is documented to return a value in [0, 0x10FFFF] (#7552).
def main() -> None:
    assert ord(chr(200)) == -61


main()
