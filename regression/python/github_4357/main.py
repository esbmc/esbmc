from collections import deque, OrderedDict


def main() -> None:
    # deque construction from a list, then list-style append.
    d = deque([1, 2, 3])
    d.append(4)
    assert len(d) == 4
    assert d[0] == 1
    assert d[3] == 4

    # Empty deque.
    e = deque()
    e.append(7)
    assert len(e) == 1

    # OrderedDict supports the dict interface.
    od = OrderedDict()
    od[1] = 10
    od[2] = 20
    assert od[1] == 10
    assert od[2] == 20
main()
