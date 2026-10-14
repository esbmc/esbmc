from collections import Counter


def test() -> None:
    c = Counter()
    c[1, 0] = 2
    c[2, 0] = 5
    c[3, 0] = 3
    top = c.most_common(1)
    assert len(top) == 1


test()
