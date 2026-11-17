# A list method on a method call receiver runs the call once (#10817).
class Gw:

    def __init__(self) -> None:
        self.calls: int = 0

    def send(self) -> list[list[str]]:
        self.calls += 1
        return [["a"], ["b"]]


def f(g: Gw) -> None:
    g.send().pop()
    g.send().append(["c"])
    g.send().clear()
    assert g.calls == 3


f(Gw())
