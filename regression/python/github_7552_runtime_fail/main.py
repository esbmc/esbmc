# ord() of a runtime chr() read one signed byte, proving ord(chr(200)) < 0
# (#7552).
def main() -> None:
    n: int = 200
    c = chr(n)
    assert ord(c) < 0


main()
