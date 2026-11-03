class C:

    def m(self, a):
        return a


c = C()
try:
    c.m(a=1, b=2)
    assert False
except TypeError:
    pass
