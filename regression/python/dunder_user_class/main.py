class Box:
    def __init__(self, n: int):
        self.n = n

    def __getitem__(self, i: int) -> int:
        return self.n + i

    def __len__(self) -> int:
        return self.n

    def __contains__(self, v: int) -> bool:
        return v < self.n


def run():
    b = Box(3)
    # Rewriting the dunder call to the operator must keep dispatching back to
    # the class's own definition.
    assert b.__getitem__(2) == 5
    assert b[2] == 5
    assert b.__len__() == 3
    assert len(b) == 3
    assert b.__contains__(1)
    assert 1 in b


run()
