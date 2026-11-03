# The class binds code itself, so a store on some paths only changes it.
class A:

    def __init__(self):
        self.code = 0


a = A()
c = nondet_bool()
if c:
    a.code = 1
assert a.code == 0
