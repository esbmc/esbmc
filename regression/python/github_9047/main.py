# A class nested in a class of an imported module, named through the module
# (#9047).
import mod

x = mod.A.B(3)
assert isinstance(x, mod.A.B)
assert x.v == 3
y = mod.A.B.C()
assert isinstance(y, mod.A.B.C)
assert not isinstance(y, mod.A.B)
