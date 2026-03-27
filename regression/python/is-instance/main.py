x = 5
if isinstance(x, int):
    assert True
else:
    assert False


class C:
    pass


class B:
    pass


class A(B):

    def __init__(self, a: int):
        self.x: int = a

    pass


y = A(10)
assert isinstance(y, B) == True
assert isinstance(y, C) == False
