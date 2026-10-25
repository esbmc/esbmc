# A class nested in a class imported from a module (#9047).
from mod import A

x = A.B.C()
assert isinstance(x, A.B)
