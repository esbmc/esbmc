# a len() result read back from a dict or list is an int (#9790)
c = {1: len("ab")}
assert isinstance(c[1], int)
assert not isinstance(c[1], bool)
l = [len((1, 2, 3))]
assert isinstance(l[0], int)
assert not isinstance(l[0], str)
