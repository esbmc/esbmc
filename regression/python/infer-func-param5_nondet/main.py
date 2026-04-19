def logical_op(a, b):
    return a and b

x = nondet_bool()
y = nondet_bool()
assert logical_op(x, y) == (x and y)
