class A:
    def __init__(self, v):
        self.ok = False
        if v:
            self.ok = True


flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
a = A(x)
assert a.ok == ((isinstance(x, int) and x != 0) or (isinstance(x, str) and x != ""))
