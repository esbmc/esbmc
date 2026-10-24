# #9126: renaming the second target must leave the lambda parameter x alone.
a = [x for x in [1, 2]]
fs = [(lambda x: x + 10)(5) for x in [1, 2]]
assert fs[0] == 15 and fs[1] == 15
