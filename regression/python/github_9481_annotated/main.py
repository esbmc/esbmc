class B:
    def __init__(self, s: set):
        self.s = s


def run(x):
    b = B(x)
    b.s.add(1)
    b.s.add(1)
    return len(b.s)


assert run({7}) == 2
