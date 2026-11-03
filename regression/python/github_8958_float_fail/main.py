f = 0.5
for i in range(3):
    try:
        a += f
    except NameError:
        a = i
assert a == 0
