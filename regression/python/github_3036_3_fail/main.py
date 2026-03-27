def main() -> None:
    xs = [1, 2, 3, 4]
    ys = [x for x in xs if x % 2 == 0]
    assert ys == [1, 4]


main()
