def f():
    class Symbol:
        def get(self):
            return 7
    x = Symbol()
    return x.get()

assert f() == 7
