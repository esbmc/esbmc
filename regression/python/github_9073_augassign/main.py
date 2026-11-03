class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)
        t += (0, )
        self._args = t


Pair(1, 3)
