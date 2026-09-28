# A method called on a constructor temporary must see the state __init__ wrote.
# Reading a data attribute off the same temporary always worked, as did calling
# the method through a named receiver; only method-call-on-temporary lost it.


class C:
    def __init__(self, v):
        self.n = v

    def size(self):
        return self.n


class D(C):
    pass


assert C(3).n == 3
c = C(3)
assert c.size() == 3
assert C(3).size() == 3
assert C(1).size() + C(2).size() == 3
assert D(4).size() == 4
