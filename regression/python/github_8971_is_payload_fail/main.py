class A(Exception):
    def __init__(self, payload):
        self.payload = payload


flag = nondet_bool()
p = nondet_int() if flag else nondet_str()
a = A(p)
assert a.payload is p
assert isinstance(a.payload, int) == (not flag)
