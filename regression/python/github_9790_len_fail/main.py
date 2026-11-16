# a len() result read back from a dict or list is an int (#9790)
l = [len((1, 2, 3))]
assert not isinstance(l[0], int)
