for i in range(2):
    try:
        a += 1
    except NameError:
        a = 0
assert a == 1
