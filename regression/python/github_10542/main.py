# #10542: an int stored into a list of bools reads back as the int.
other = [False, False]
other[0] = 5
assert other[0] == 5
assert type(other[0]) is int
assert other[1] == False

rep = [False] * 2
rep[0] = 7
assert rep[0] == 7
