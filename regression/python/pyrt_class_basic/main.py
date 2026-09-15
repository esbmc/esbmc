class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self, step):
        self.value = self.value + step
        return self.value


c = Counter(3)
assert c.bump(2) == 5
assert c.value == 5
Counter.limit = 10
assert c.limit == 10
c.limit = 1
assert c.limit == 1 and Counter.limit == 10
