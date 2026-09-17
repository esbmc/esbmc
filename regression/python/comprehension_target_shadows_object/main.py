# Python 3 scopes a comprehension's target to the comprehension. Sharing one
# symbol with an outer variable of a different type aborted the converter in
# member2t once the outer object was rebound to elements.


class C:
    def __init__(self, xs=[]):
        self.xs = xs


def main():
    c = C([1, 2])

    node = c
    out = [node for node in node.xs]
    assert len(out) == 2
    assert (out[0]) == 1

    node2 = c
    gen = list(node2 for node2 in node2.xs)
    assert len(gen) == 2

    node3 = c
    acc = []
    acc.extend(node3 for node3 in node3.xs)
    assert len(acc) == 2

    xs = [1, 2, 3]
    item = xs
    same = [item for item in item]
    assert len(same) == 3


main()
