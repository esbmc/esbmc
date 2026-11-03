# A from-import binds its name over a plain import of another module (#9046)
import one
from two import f

assert f() == 1
