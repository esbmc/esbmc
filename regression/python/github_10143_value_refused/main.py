# #10143: the value of xs and True is a list or a bool.
xs = [1]
y = xs and True
assert y
