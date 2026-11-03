class A(Exception):
    def __init__(self, payload):
        self.payload = payload


flag = nondet_bool()
p = nondet_int() if flag else nondet_str()
a = A(p)
assert not (a.payload is not p)
if a.payload is not p:
    assert False
