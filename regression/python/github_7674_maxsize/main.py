import sys


def main() -> None:
    assert sys.maxsize == 9223372036854775807
    assert sys.byteorder == "little"


main()
