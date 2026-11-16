# Issue #9642: raise C calls C(), which needs an argument here.
class E(Exception):
    def __init__(self, code: int):
        self.code = code

b = 0
try:
    raise E
except E:
    b = 1
assert b == 1
