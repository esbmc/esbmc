x = 10


def modify():
    global x
    x = x + 1


modify()
assert (x == 11)
