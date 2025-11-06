class Base:
    data: int = 1  # Class attribute 'data' with default value 1
    x: int  # Declare instance attribute
    y: int  # Declare instance attribute

    def __init__(self, x: int) -> None:  # Add return type annotation
        self.x: int = x

    def foo(self, y: int) -> None:  # Add return type annotation
        self.y: int = y

    def bar(self) -> int:
        return 1


class Middle(Base):
    data: int = 2  # Overriding class attribute 'data' in the middle class

    def bar(self) -> int:  # Overriding 'bar' method in the middle class
        return 2
    
    def baz(self) -> int:  # New method in middle class
        return 10


class Derived(Middle):
    data: int = 3  # Overriding class attribute 'data' in the derived class

    def bar(self) -> int:  # Overriding 'bar' method again in the derived class
        return 3


a: int = 10
obj: Derived = Derived(a)
assert obj.x == 10  # Asserting that 'x' attribute is set correctly after Base::__init__ call

b: int = 5
obj.foo(b)  # Calling base class method 'foo' with argument 'b'
assert obj.y == 5  # Asserting that 'y' attribute is set correctly in the base class method 'foo'

assert Base.data == 1  # Asserting that the base class attribute 'data' is unchanged
assert Middle.data == 2  # Asserting that the middle class has its own 'data' attribute
assert Derived.data == 3  # Asserting that the derived class has its own 'data' attribute

c: int = obj.bar()  # Testing the overridden 'bar' method in the derived class
assert c == 3  # Asserting that the overridden 'bar' method returns the expected value (3)

d: int = obj.baz()  # Testing inherited method from middle class
assert d == 10  # Asserting that the inherited 'baz' method returns the expected value (10)


class MyFloat(float):
    pass