class R:
    def __init__(self) -> None:
        self.r = None

    def ready(self) -> bool:
        return self.r is not None


o = R()
assert not o.ready()
assert o.r is None
