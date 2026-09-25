class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self, step):
        self.value = self.value + step
        return self.value


c = Counter(3)
assert c.bump(2) == 6
