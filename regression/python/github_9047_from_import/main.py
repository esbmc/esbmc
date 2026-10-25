# A class nested in a class imported from a module (#9047).
from mod import A
from mod import A as Outer

x = A.B(3)
assert isinstance(x, Outer.B)
assert x.v == 3
assert isinstance(Outer.B.C(), A.B.C)
