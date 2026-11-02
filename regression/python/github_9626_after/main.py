# The comprehension variable has the name of a variable bound after it.
e = [i * 2 for i in [1.5, 2.5]]
i = {1: 2.0}
p = i[1]
assert p == 2.0
assert e == [3.0, 5.0]
