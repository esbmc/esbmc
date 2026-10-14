def main():
    xs = [1, 2, 3]
    xs += [4]
    assert len(xs) == 4
    assert (xs[3]) == 4

    ys = [1, 2]
    ys += [3, 4]
    assert (ys[0]) == 1
    assert (ys[2]) == 3
    assert (ys[3]) == 4


main()
