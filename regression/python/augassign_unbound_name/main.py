try:
    a += 1
except NameError:
    a = 1
assert a == 1
