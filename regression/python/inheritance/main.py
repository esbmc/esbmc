class Base:
    def __init__(self, x:int):
        self.x:int = x

    def foo(self):
        pass

class Derived(Base):
    pass

a = 10
obj = Derived(a)
assert obj.x == 10
obj.foo() # Calling base class method
