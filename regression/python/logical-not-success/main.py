x: int = 1
y: int = 2
z: int = 3

if (not (x == 2) and not (y == 2) and z == 3):
    result: int = 1 / 0


class MyClass:

    def __init__(self, v: bool):
        self.value = v


obj = MyClass(False)
assert (not obj.value) == True
