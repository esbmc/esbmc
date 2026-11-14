class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val


cond = nondet_bool()
v = Cell(A()) if cond else Cell(2)
x = v.value + 23
assert x == 25
