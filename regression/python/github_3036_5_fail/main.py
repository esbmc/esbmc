def main() -> None:
    xs = [1, 2, 3]
    ys = [x + 0.5 for x in xs]
    assert ys == [1.5, 2.5, 3.4]


main()
