# A tagged element compares equal to the plain element it holds, in both
# directions, so the tagged path is not answering "unequal" for everything.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
a = [x]
if cond:
    assert a == [1]
else:
    assert a == ["a"]
