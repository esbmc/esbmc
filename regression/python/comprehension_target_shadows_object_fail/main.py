class C:
    def __init__(self, xs=[]):
        self.xs = xs


def main():
    c = C([1, 2])
    node = c
    out = [node for node in node.xs]
    assert len(out) == 5


main()
