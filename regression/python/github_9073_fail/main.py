class And:

    def __init__(self, *args):
        self._args = args

    def first(self) -> int:
        return self._args[0] // self._args[1]


a = And(nondet_int(), 0)
a.first()
