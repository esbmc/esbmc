# #10109: a None-returning call stored into a name that otherwise holds an
# instance.
class C:
    def __init__(self):
        self.v = 1


def f():
    return C()


def g():
    pass


def run(flag):
    if flag:
        r = f()
    else:
        r = g()
    if flag:
        assert r is not None
    else:
        assert r is not None


run(True)
run(False)
