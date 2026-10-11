def check(v) -> bool:
    return isinstance(v, int | str)


assert isinstance(1, int | str)
assert not isinstance(1.5, int | str)
assert isinstance("a", str | None)
assert isinstance(2.5, (int | str, float))
assert check(3)
assert not check(1.0)
assert issubclass(bool, str | int)
assert not issubclass(float, int | str)
