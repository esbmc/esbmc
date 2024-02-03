class Base:
    data: int = 1  # Class attribute 'data' with default value 1

    def __init__(self, x: int):
        self.x: int = x

    def foo(self, y: int):
        self.y: int = y

    def bar(self) -> int:
        return 1


class Derived(Base):
    data: int = 2  # Overriding class attribute 'data' in the derived class

    def bar(self) -> int:  # Overriding 'bar' method in the derived class
        return 2


a = 10
obj = Derived(a)
assert obj.x == 10  # Asserting that 'x' attribute is set correctly after Base::__init__ call

b: int = 5
obj.foo(b)  # Calling base class method 'foo' with argument 'b'
assert obj.y == 5  # Asserting that 'y' attribute is set correctly in the base class method 'foo'

assert Base.data == 1  # Asserting that the base class attribute 'data' is unchanged
assert Derived.data == 2  # Asserting that the derived class has its own 'data' attribute

c: int = obj.bar()  # Testing the overridden 'bar' method in the derived class
assert c == 2  # Asserting that the overridden 'bar' method returns the expected value (2)
