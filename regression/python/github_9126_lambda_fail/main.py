# #9126
a = [x for x in [1, 2]]
fs = [(lambda x: x + 10)(5) for x in [1, 2]]
assert fs[0] == 11
