class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val


cond = nondet_bool()
v = Cell(A()) if cond else Cell(2)
if not cond:
    x = v.value + 23
    assert x == 25
