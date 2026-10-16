class A:
    def __init__(self, m):
        self.m = m
        self.n = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.n >= self.m:
            raise StopIteration
        self.n += 1
        return self.n

class B:
    def __init__(self, m):
        self.m = m
        self.n = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.n >= self.m:
            raise StopIteration
        self.n += 1
        return 10 * self.n

def total(it):
    t = 0
    for i in it:
        t += i
    return t

assert total(A(2)) == 3
assert total(B(2)) == 30
