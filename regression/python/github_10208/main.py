# Issue #10208: bytes() of a list held in a variable relabelled the list
# pointer as a byte array, giving a wrong length (and a false proof of
# len(b) == 1). Its size is not known statically, so it is refused.
buf = [0, 0]
b = bytes(buf)
assert len(b) == 2
