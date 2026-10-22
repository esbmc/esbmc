class D:
    def __init__(self):
        self.cells = [0, 1, 2]

    def drop(self, v):
        self.cells.remove(v)

    def echo(self, v):
        return v


class E(D):
    pass


d = D()
d.drop(1)
assert d.cells == [0, 2]

# Calls disagreeing on the type leave the parameter unannotated.
assert d.echo(1) == 1
assert d.echo("x") == "x"

e = E()
e.drop(2)
assert e.cells == [0, 1]
