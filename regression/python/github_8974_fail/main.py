# An attribute set to a str in __init__ and later assigned an int (#8974).
class A:

    def __init__(self):
        self.b = "s"


a = A()
a.b = 1
assert a.b != 1
