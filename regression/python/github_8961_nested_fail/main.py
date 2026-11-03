# Calling the result of a call of a nested function (#8961).
def h():
    return 2


def func():

    def g():
        return h

    return g()()


assert func() == 3
