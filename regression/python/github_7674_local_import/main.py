def bounded() -> float:
    import sys
    return sys.float_info.max


def main() -> None:
    assert bounded() > 0.0


main()
