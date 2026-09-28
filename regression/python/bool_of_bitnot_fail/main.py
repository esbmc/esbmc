# bool() of an int expression converts its value (x != 0).
k: int = 0
assert not bool(~k)
