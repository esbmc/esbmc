class And:

    def __init__(self, *args):
        self._args = args

    def first(self) -> int:
        return self._args[0] // self._args[1]


a = And(1, 2)
a.__class__(1, 0, 3).first()
