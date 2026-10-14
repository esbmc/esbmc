# Flow-sensitive class tracking (#4772): when a variable is reassigned to a
# different class (`n1 = A(1); n1 = B()`), the usage-site scanner still drops
# the lexical class mapping, but flow_class_map_ tracks the LAST write per
# variable at unconditional top-level scope. So at `a.x = n1`, n1's current
# class is B, and the nested read `a.x.data` casts to B and resolves — matching
# CPython. Gated to straight-line depth-1 code and cleared across control-flow
# joins so a class is never adopted unsoundly.

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
