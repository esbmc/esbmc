class uint64(int):
    pass


class Gwei(uint64):
    pass


SOME_CONST = Gwei(5)


class Validator:
    field: Gwei


v = Validator()
v.field = Gwei(5)
assert v.field == SOME_CONST
