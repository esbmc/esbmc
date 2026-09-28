# len() of a list element dispatches the class's __len__ (#7966).
count = 0


class C:

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        global count
        count += 1
        return self.n


xs = [C(2), C(1)]
k = len(xs[0])
assert count == 0
