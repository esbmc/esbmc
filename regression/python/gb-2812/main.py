class Foo:
    def __init__(self, x: int):
        self.x = x

def foo(x: int) -> Foo:
    return Foo(x)

f = foo(4)
