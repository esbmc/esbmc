# 10464: max() over a named mixed list returns the int element.
l = [2.5]
l.append(3)
m = max(l)
assert m == 3 and type(m) is int
