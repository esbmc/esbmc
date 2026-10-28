class And:

    def __init__(self, *args):
        self._args = args

    def first(self) -> int:
        return self._args[0] // self._args[1]


x = nondet_int()
a = And(x, 2)
assert a.first() == x // 2
