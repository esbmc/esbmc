class Base:
    def __init__(self, x:int):
        self.x:int = x

    def foo(self, y:int):
        self.y:int = y

class Derived(Base):
    pass

a = 10
obj = Derived(a)
assert obj.x == 10  # Base::__init__ adds x attribute

v:int = 5
obj.foo(v) # Calls base class method
assert obj.y == 6 # This assert fails
