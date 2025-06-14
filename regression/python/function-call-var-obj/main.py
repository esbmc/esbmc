class MyClass:
    lucas: int = 1

def return_obj(y: MyClass) -> MyClass:
    return y

z: MyClass = MyClass()
z.lucas = 3
obj2: MyClass = return_obj(z)
assert obj2.lucas == 3

