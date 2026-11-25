k = lambda: 0
if nondet_bool():
    k = lambda: 1
assert k.__name__ == "<lambda>"
