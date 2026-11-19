class Gw:
    def __init__(self) -> None:
        self.items: list[list[str]] = [["a"], ["b"]]

    def send(self) -> list[list[str]]:
        return self.items


def f(g: Gw) -> None:
    g.send().pop()
    g.send().append(["c"])
    assert len(g.items) == 2
    assert g.items[1][0] == "c"


f(Gw())
