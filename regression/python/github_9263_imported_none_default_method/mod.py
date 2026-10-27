class C:
    def func(self, x=None):
        if x is None:
            return "a"
        return x

obj = C()
