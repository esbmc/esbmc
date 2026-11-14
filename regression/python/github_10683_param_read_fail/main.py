# show reads its parameter as a value, so the int it is passed stays an int.
def g():
    return 1


def show(a):
    print(a)
    return a


def pick():
    return show


pick()(7)()
