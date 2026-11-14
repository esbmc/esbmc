mixed = [1, True]
assert mixed[0] == 1 and mixed[1] == True
other = [False]
other.append(5)
assert other[1] == 5 and other[0] == False
