def mixed_statements() -> int:
    x: int = 1
    pass
    y: int = 2
    return x + y


result: int = mixed_statements()
assert result == 3


class MixedClass:
    x: int = 5
    pass
    y: int = 10


assert MixedClass.x == 5
assert MixedClass.y == 10
