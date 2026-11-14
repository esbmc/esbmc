# The int get returns is also read, so it must stay an int: x + 1 is 8, and
# get(False)() raises TypeError.
def g():
    return 1


def get(flag):
    if flag:
        a = g
    else:
        a = 7
    return a


x = get(False)
assert x + 1 == 8
get(False)()
