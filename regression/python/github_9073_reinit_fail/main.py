class And:

    def __init__(self, *args):
        self._args = args

    def first(self) -> int:
        return self._args[0] // self._args[1]


a = And(1, 2)
try:
    a.__init__(1, 0, 3)
except TypeError:
    pass
a.first()
