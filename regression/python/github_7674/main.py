import sys


def main() -> None:
    assert sys.float_info.max > 0.0
    assert sys.float_info.epsilon > 0.0
    assert sys.float_info.mant_dig == 53


main()
