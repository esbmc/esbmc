class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)
        if a > b:
            t = (a, b, 0)
        self._args = t


Pair(1, 3)
