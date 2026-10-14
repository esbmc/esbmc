class Service:

    def __init__(self, name):
        self._name = name
        self._tag = 1


g = 0


def make() -> None:
    global g
    g = Service("hi")


make()
