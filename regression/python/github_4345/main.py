def main() -> None:
    # list.sort(reverse=True)
    xs = [3, 1, 2]
    xs.sort(reverse=True)
    assert xs[0] == 3 and xs[1] == 2 and xs[2] == 1

    # list.sort with no args still ascending.
    ys = [3, 1, 2]
    ys.sort()
    assert ys[0] == 1 and ys[2] == 3

    # sorted(reverse=True) over a literal list.
    zs = sorted([3, 1, 2], reverse=True)
    assert zs[0] == 3 and zs[2] == 1

    # sorted(reverse=True) over a list variable.
    src = [10, 30, 20]
    out = sorted(src, reverse=True)
    assert out[0] == 30 and out[2] == 10

    # sorted with no kwargs still ascending.
    asc = sorted([5, 2, 4])
    assert asc[0] == 2
main()
