class W:
    def __init__(self):
        self.tab = [None, 1]

    def second(self):
        return self.tab[1]


assert W().second() == 2
