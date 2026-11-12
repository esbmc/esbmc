other = [False, False]
other[0] = 5
assert other[0] == 5
assert other[1] == False

rep = [False] * 2
rep[0] = -3
assert rep[0] == -3
assert rep[1] == False

flags = [True, False]
flags[1] = 7
assert type(flags[1]) is int
assert flags[0] + flags[1] == 8
