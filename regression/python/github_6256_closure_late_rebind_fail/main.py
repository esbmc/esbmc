# A closure reads the enclosing binding, not a def-time copy: g() is 2, not the
# 1 that x held when `def g` ran. A capture cell frozen at the def would prove
# this false claim, so the cell is only created when nothing rebinds the name
# after the def (#6256).


def f():
    x = 1

    def g():
        return x

    x = 2
    return g


h = f()
assert h() == 1
