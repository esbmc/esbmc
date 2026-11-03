class C:

    def func(self):
        return 1


ok = False
try:
    C.func()
except TypeError:
    ok = True
assert ok
