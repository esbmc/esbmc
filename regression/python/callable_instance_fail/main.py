# __call__ returns x + 1, so c(4) is 5, not 99.
class C:
    def __call__(self, x):
        return x + 1


c = C()
assert c(4) == 99
