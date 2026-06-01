from collections import defaultdict


def test() -> None:
    d = defaultdict(list)
    d[1].append(10)
    # The first element is 10, not 11 — assertion should fail.
    assert d[1][0] == 11


test()
