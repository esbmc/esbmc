from collections import Counter


def test() -> None:
    c = Counter()
    c[1, 0] = 2
    c[2, 0] = 5
    # The top count is 5, not 2 — assertion should fail.
    top = c.most_common(1)
    assert top[0] == 2


test()
