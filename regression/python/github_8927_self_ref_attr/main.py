class A:
    def __init__(self):
        self.ref = self
        self.n = 0

    def me(self):
        return self.ref

    def other(self, b):
        self.ref = b


a = A()
r = a.ref
r.t = 1
assert a.t == 1
s = a.me()
s.n = 3
assert a.n == 3

b = A()
b.n = 7
a.other(b)
assert a.ref.n == 7


# A method named like str.find no longer reads `ref` as a str (#8926).
class Sq:
    def __init__(self, pos):
        self.pos = pos
        self.t = 0
        self.ref = self

    def find(self, update=False):
        ref = self.ref
        if ref.pos != self.pos:
            ref = ref.find(update)
        return ref


q = Sq(0)
w = q.find()
w.t = 1
assert q.t == 1
