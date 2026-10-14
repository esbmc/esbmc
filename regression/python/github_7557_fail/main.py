# int("10", base=2) is 2, not the 10 the dropped keyword used to prove (#7557).
def main() -> None:
    assert int("10", base=2) == 10


main()
