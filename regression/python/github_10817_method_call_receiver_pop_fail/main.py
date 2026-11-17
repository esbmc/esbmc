# pop() on the empty list a method call returns raises IndexError (#10817).
class Gw:
    def send(self) -> list[list[str]]:
        return []

def f(g: Gw) -> None:
    g.send().pop()

f(Gw())
