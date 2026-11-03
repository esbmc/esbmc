# The base class binds code, so a store on some paths only changes it.
class A:

    def __init__(self):
        self.code = 0


class B(A):
    pass


b = B()
c = nondet_bool()
if c:
    b.code = 1
assert b.code == (1 if c else 0)
