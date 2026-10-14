def main() -> None:
    # max() over a dict is the max key (3), not 1.
    d = {3: 1, 1: 2, 2: 3}
    assert max(d) == 1


main()
