# A decorator defined in a function body is refused rather than ignored
# (#8960).
replace = True


def func():

    def dec(f):
        return other if replace else f

    def other():
        return 7

    @dec
    def inner():
        return 1

    return inner()


assert func() == 7
