class A:
    class B:
        def __init__(self, a):
            self.a = a

        def bfunc(self):
            return 42


class C(A.B):
    def __init__(self, a):
        super().__init__(a)

    def cfunc(self):
        return self.a


flag = nondet_bool()
x = nondet_str() if flag else nondet_int()
c = C(x)
d = c.cfunc()
assert d is x
assert isinstance(d, str) == flag
assert c.bfunc() == 42
