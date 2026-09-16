i = 0
x = 0
while i < 1000000:
    assert x == 0
    x = 1 - x
    i = i + 1
