import sys


def main() -> None:
    assert sys.float_info.max == 1.7976931348623157e+308
    assert sys.float_info.min == 2.2250738585072014e-308
    assert sys.float_info.epsilon == 2.220446049250313e-16
    assert sys.float_info.mant_dig == 53
    assert sys.float_info.radix == 2


main()
