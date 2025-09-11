def func() -> int:
    return 2


def get_negative() -> int:
    return -5


def is_positive(x: int) -> bool:
    return x > 0


def get_true() -> bool:
    return True


def get_false() -> bool:
    return False


def logical_and(a: bool, b: bool) -> bool:
    return a and b


assert is_positive(func()) == True
assert is_positive(get_negative()) == False
assert logical_and(get_true(), get_false()) == False
