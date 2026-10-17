class W:
    def __init__(self):
        self.tab = [None, 1]

    def find(self, i):
        return self.tab[i]

assert W().find(1) == 1
