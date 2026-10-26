class M:
    def f(self):
        return 1

    def get(self):
        return self.f


a = M()
g = a.get()
assert g() == 1
