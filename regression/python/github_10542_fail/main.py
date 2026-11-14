# #10542: an int stored into a list of bools reads back as the int.
other = [False, False]
other[0] = 5
assert other[0] == 1
