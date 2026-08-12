# An attribute may be read straight off the value an operator produces, not
# just off a name: `(a + b).x` used to abort with
# "Unsupported Attribute value type: BinOp" while `c = a + b; c.x` worked.


class V:
    def __init__(self, x: int):
        self.x = x

    def __add__(self, o: "V") -> "V":
        return V(self.x + o.x)

    def __neg__(self) -> "V":
        return V(-self.x)


a = V(1)
b = V(2)

assert (a + b).x == 3
assert (-a).x == -1
assert (V(4) + V(5)).x == 9
assert (a + b + a).x == 4

c = a + b
assert c.x == 3
