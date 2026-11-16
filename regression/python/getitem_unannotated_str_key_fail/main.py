class Table:
    def __init__(self):
        self.d = {"a": 1, "b": 2}

    def __getitem__(self, k):
        return self.d[k]


t = Table()
assert t["b"] == 1
