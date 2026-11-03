# #8915: `neighbour` is read from a list attribute filled by append, so its
# class is not known and the call to the user method head() is refused.
class Square:
    def __init__(self):
        self.timestamp = 1
        self.neighbours = []

    def head(self):
        return self


def useful(square):
    for neighbour in square.neighbours:
        r = neighbour.head()
        r.timestamp = 0


a = Square()
a.neighbours.append(a)
useful(a)
assert a.timestamp == 0
