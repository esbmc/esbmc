# len() of an element whose class has no __len__ raises TypeError (#7966).
class D:

    def __init__(self, n: int):
        self.n = n


ds = [D(1)]
k = len(ds[0])
