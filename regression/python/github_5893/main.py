def choices():
    return (1, 2)


x = 3
if x in choices():
    assert x == 1 or x == 2
