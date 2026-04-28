# Known limitation: when a variable is reassigned to a different class,
# the usage-site scanner conservatively drops the variable's class mapping
# to avoid unsoundly attributing later attribute writes to the wrong
# class. This is safer than the pre-fix behaviour (which would silently
# adopt the shadowing class's type) but means legitimate code that mixes
# reassignment with attribute writes can't benefit from type inference.
#
# Recovering this case would require flow-sensitive (per-program-point)
# class tracking rather than the current lexical scan.

class A:
    def __init__(self, v):
        self.value = v
        self.x = None


class B:
    def __init__(self):
        self.data = 99


n1 = A(1)     # var_to_class[n1] = "A"
n1 = B()      # conflict — scanner drops var_to_class[n1]

a = A(2)
a.x = n1      # scanner can't resolve n1's class, so .x stays any_type()

# Python runtime: a.x is the B instance, so .data access yields 99.
# ESBMC today: .x typed as void*, nested access aborts before reaching here.
assert a.x.data == 99
