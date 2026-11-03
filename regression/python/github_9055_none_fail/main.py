# #9055: None + 1 raises TypeError.
class A:
    def f(self):
        return 1


def f(x):
    if x > 0:
        res = A().f()
    else:
        res = None
    return res


a = f(0)
b = a + 1
