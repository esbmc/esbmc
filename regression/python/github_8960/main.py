# A nested function returns a sibling defined after it (#8960).
def func():

    def g():
        return h

    def h():
        return 2

    k = g()
    return k()


assert func() == 2
