class V:
    def __init__(self, a):
        self.a = a

    def __add__(self, o):
        return V(self.a + o.a)

    def __eq__(self, o):
        return self.a == o.a


r = V(2) + V(3)
assert r.a == 5

chained = V(1) + V(2) + V(4)
assert chained.a == 7

assert V(6) == V(6)

acc = V(10)
acc = acc + V(5)
assert acc.a == 15
