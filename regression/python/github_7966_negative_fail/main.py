# A negative __len__ result makes len() raise ValueError (#7966).
class C:

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


xs = [C(-1)]
k = len(xs[0])
