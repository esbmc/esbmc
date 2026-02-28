def main() -> None:
    xs = [1, 2, 3]
    ys = [x * 2 for x in xs]
    assert ys == [2, 4, 6]


main()
