# The element-class map is keyed by bare variable name across the whole module,
# so it must be conservative: a name that is a class instance in one scope and a
# plain value in another must not be force-typed as the class (which would
# mis-store the loop element). Here `a` is an A() in f() but an int in g();
# g()'s loop must keep it an int (#4805 review finding).


class A:
    def __init__(self):
        self.x = 7


def f():
    a = A()
    return a.x


def g():
    a = 3
    total = 0
    for v in [a]:
        total = total + v
    return total


assert f() == 7
assert g() == 3
