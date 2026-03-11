class A:
    pass

class B(A):
    def f(self) -> str:
        return "Woof!"

def test():
    b = B()
    x: int = b.f()
    assert x == "Woof!"

test()
