def main() -> None:
    # By key=abs the max is -5 (abs 5), not 3; this assertion must fail.
    assert max([1, -5, 3], key=abs) == 3


main()
