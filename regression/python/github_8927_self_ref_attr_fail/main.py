class A:
    def __init__(self):
        self.ref = self

a = A()
r = a.ref
r.t = 1
assert a.t == 2
