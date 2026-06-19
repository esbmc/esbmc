# Negative variant of class_as_value: passing the class object as a bare
# argument converts, but the assertion on a real instance does not hold.
# Confirms modelling the class-as-value as a placeholder does not vacuously
# accept claims about instances built from the same class.

class A:
    def __init__(self):
        self.x = 7


def register(cls):
    return 0


register(A)

a = A()
assert a.x == 99
