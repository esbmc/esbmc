class MyClass:
    def __init__(self, value: int):
        self.x:int = value

    def foo(self):
        pass


obj1 = MyClass(5)
assert(obj1.x == 5)
obj1.x = 10
assert(obj1.x == 10)
obj1.x = 20
assert(obj1.x == 20)

obj2 = MyClass(30)
assert(obj2.x == 30)
obj2.x = 40
assert(obj2.x == 40)

obj2.foo() # Calling base class method
