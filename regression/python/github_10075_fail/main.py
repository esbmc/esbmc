# #10075: the int path of an int-or-None call result must reach the assert.
def f():
    return 42


def g():
    pass


def run(flag):
    if flag:
        r = f()
    else:
        r = g()
    if flag:
        assert r != 42


run(True)
