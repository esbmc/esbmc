def f():
    return g()


def g():
    return "global"


def test():

    def g():
        return "local"

    return g()


assert f() == "global"
assert test() == "local"
