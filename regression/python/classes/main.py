class MyClass:
    # Class attribute
    class_attr:int = 25

    def __init__(self, value: int):
        self.data:int = value


obj1 = MyClass(5)
assert(obj1.data == 5)
obj1.data = 10
assert(obj1.data == 10)

obj2 = MyClass(15)
obj2.data = 20
assert(obj2.data == 20)

assert(MyClass.class_attr == 25)

obj3 = MyClass(30)
obj3.class_attr = 35 # Instance attributes can be set in a method with self.name = value
assert(obj3.class_attr == 35) # Instance attribute hides a class attribute with the same name
assert(MyClass.class_attr == 25)

MyClass.class_attr = 50
assert(MyClass.class_attr == 50)
assert(obj3.class_attr == 35)

assert(obj1.class_attr == 50) # Class attribute might be referred as obj1 don't have instance attributes
assert(obj2.class_attr == 50) # Class attribute might be referred as obj2 don't have instance attributes
