class MyClass:
    def __init__(self, value: int):
        self.data:int = value

    def foo(self) -> None:
        self.data = 1

    def blah(self) -> None:
        x:int = self.data


obj1 = MyClass(5)
assert(obj1.data == 5)
obj1.data = 10
assert(obj1.data == 10)
obj1.data = 20
assert(obj1.data == 20)

obj2 = MyClass(30)
assert(obj2.data == 30)
obj2.data = 40
assert(obj2.data == 40)
