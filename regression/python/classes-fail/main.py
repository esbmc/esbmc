class Foo:

    def __init__(self, value: int):
        self.blah: int = value
        x: float = 1 / self.blah


f = Foo(0)
