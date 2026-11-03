class S:
    def __call__(self, u):
        return u + 1


s = S()
x = s(0)
assert x == 2
