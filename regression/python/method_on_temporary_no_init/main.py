# A class with no __init__ anywhere in its MRO has no constructor to call, so
# the temporary receiver is an uninitialised instance -- a method that reads no
# state still works.


class C:
    def value(self):
        return 7


class D(C):
    pass


assert C().value() == 7
assert D().value() == 7
