# A tag only ever holds a bool, int, float or str, so an aggregate or a user
# class can never match. Issue #7075: these used to abort the frontend.


class C:
    def __init__(self) -> None:
        self.v: int = 0


cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert not isinstance(x, list)
assert not isinstance(x, dict)
assert not isinstance(x, tuple)
assert not isinstance(x, C)
assert isinstance(x, object)
