# The attribute carries the operator's real result, so asserting the wrong
# value stays refutable.


class V:
    def __init__(self, x: int):
        self.x = x

    def __add__(self, o: "V") -> "V":
        return V(self.x + o.x)


a = V(1)
b = V(2)
assert (a + b).x == 4
