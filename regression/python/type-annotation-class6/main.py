class A:

    def __init__(self, x, name):
        self.x = x
        self.name = name


a = A(10, "bob")
assert a.x == 10 and a.name == "bob"
