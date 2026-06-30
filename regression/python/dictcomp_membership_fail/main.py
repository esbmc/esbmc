def main() -> None:
    # 5 is not a key of {0: 0, 1: 1, 2: 4}, so `5 in d` is False.
    d = {x: x * x for x in range(3)}
    assert 5 in d


main()
