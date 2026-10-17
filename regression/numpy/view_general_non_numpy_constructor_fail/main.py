class Factory:
    def eye(self, n):
        return n + 1

    def full(self, n, x):
        return n * x

f = Factory()
assert f.eye(3) == 4
assert f.full(2, 3) == 7
