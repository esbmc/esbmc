# 9829: an unbound += of a generator's loop variable under except NameError
def gen():
    yield 1
    yield 2


for x in gen():
    try:
        a += x
    except NameError:
        a = x
assert a == 4
