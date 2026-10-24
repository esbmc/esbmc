# #8915: head() writes to its receiver. Lowering the unresolved call to a
# null constant dropped that write and proved the final assertion, which
# CPython fails.
class Square:
    def __init__(self):
        self.timestamp = 1
        self.neighbours = []

    def head(self):
        self.timestamp = 5
        return 1


def useful(square):
    for neighbour in square.neighbours:
        r = neighbour.head()


a = Square()
a.neighbours.append(a)
useful(a)
assert a.timestamp == 1
