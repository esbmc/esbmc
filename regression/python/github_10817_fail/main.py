class Gw:
    def send(self) -> list[list[str]]:
        return []


def f(g: Gw) -> None:
    g.send().pop()


f(Gw())
