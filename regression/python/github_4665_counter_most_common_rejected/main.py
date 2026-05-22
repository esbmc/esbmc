from collections import Counter


def test() -> None:
    c = Counter([1, 1, 2])
    top = c.most_common(1)
    assert top[0][0] == 1


test()
