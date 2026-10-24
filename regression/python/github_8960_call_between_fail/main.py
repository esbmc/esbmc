# A call between the two defs runs before the later sibling is bound, so
# CPython raises NameError (#8960).
def func():

    def g():
        return h

    k = g()

    def h():
        return 2

    return k()


assert func() == 2
