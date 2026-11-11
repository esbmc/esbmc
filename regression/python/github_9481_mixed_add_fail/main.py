class B:
    def __init__(self, s):
        self.s = s


a = B(set())
b = B([])
b.s.add(1)
