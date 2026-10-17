class W:
    def __init__(self):
        self.tab = [None, 1]

    def second(self):
        return self.tab[1]

    def first(self):
        return self.tab[0]


assert W().second() == 1
assert W().first() is None
