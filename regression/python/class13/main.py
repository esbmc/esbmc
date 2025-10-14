class MyClass:
    class_attr: int = 1
    mutable_attr: list = []

    def __init__(self, value: int):
        self.data: int = value

class ChildClass(MyClass):
    child_attr: int = 100

    def __init__(self, value: int, child_value: int = 50):
        super().__init__(value)
        self.child_data = child_value

assert MyClass.class_attr == 1
obj1 = MyClass(10)
obj1.class_attr = 2
assert obj1.class_attr == 2

obj2 = ChildClass(20)
assert obj2.child_attr == 100
