class Square:
    def __init__(self):
        self.timestamp = 1
        self.neighbours = [self]

    def head(self):
        self.timestamp = 5
        return 1


def useful(square):
    for neighbour in square.neighbours:
        r = neighbour.head()


a = Square()
useful(a)
assert a.timestamp == 1
