# E binds code itself, so a store on some paths only changes it.
class E(Exception):

    def __init__(self):
        self.code = 0


a = E()
c = nondet_bool()
if c:
    a.code = 1
assert a.code == (1 if c else 0)
