def choices():
    return (1, 2)


x = 2
if x in choices():
    assert x == 1
