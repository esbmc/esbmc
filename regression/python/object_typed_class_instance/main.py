# An `object`/Any-typed variable assigned a class instance: the migration must
# allocate the instance as that class (not a void pointer) and the variable
# holds it.
class Box:
    def __init__(self, v: int) -> None:
        self.v = v


t: object = Box(7)
assert t.v == 7
