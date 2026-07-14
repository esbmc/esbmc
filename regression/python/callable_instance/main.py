# Calling an instance whose class defines __call__ (c(args)) previously crashed
# ESBMC (to_code_type on the instance's class-struct type). It is now dispatched
# to c.__call__(args). (A temporary receiver like C()() is a separate case, not
# covered here.)
class C:
    def __call__(self, x):
        return x + 1


c = C()
assert c(4) == 5
assert c.__call__(4) == 5          # explicit form matches


class NoArg:
    def __call__(self):
        return 7


n = NoArg()
assert n() == 7


# A stateful callable object.
class Adder:
    def __init__(self, num):
        self.num = num

    def __call__(self, x):
        return x + self.num


add5 = Adder(5)
assert add5(10) == 15
assert add5(0) == 5
