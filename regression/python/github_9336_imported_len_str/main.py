# Control: the imported function returns a str, which has a length.
from helper import g

c = g()
assert len(c) == 2
