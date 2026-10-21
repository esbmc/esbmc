for i in range(3):
    try:
        a += i
    except NameError:
        a = i
assert a == 3
