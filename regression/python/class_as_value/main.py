# A class name used as a first-class value: passed as a bare argument (the class
# object itself, not an instance). Python classes are objects, so `register(A)`
# forwards the class. The frontend previously aborted with "Variable 'A' is not
# defined" because a class Name was neither a variable nor a function. It is now
# modelled as an opaque placeholder for inert uses, so constructing instances
# normally still works.

class A:
    def __init__(self):
        self.x = 7


class B:
    def __init__(self):
        self.y = 9


def register(cls):
    # The class object is stored/forwarded but not constructed here.
    return 0


register(A)
register(B)

# Real construction through the class name is unaffected.
a = A()
b = B()
assert a.x == 7
assert b.y == 9
