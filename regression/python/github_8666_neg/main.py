class W:
    def __init__(self):
        self.tab = [None, 1]

    def last(self):
        return self.tab[-1]


assert W().last() == 1
