from collections import defaultdict


def test() -> None:
    d = defaultdict(list)
    # Missing-key access invokes the list() factory and the resulting list
    # accepts in-place mutations through the dict slot.  Before #4356 the
    # factory was ignored and the first append failed with "List variable
    # not found".
    d[1].append(10)
    assert d[1][0] == 10


test()
