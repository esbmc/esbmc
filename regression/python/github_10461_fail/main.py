# 10461: the issue's program, the last element is an int.
l = [0.5]
l.append(3)
assert type(l[-1]) is float
