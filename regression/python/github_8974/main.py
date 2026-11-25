# An attribute set to a str in __init__ and later assigned an int (#8974).
class A:

    def __init__(self):
        self.b = "s"


def run(c):
    a = A()
    if c:
        a.b = 1
    v = a.b
    return v == 1


a = A()
a.b = 1
assert a.b == 1
assert run(True)
assert not run(False)
