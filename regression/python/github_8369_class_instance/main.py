# A global bound to a class instance is registered by its binding alone,
# so a read converted before that binding is refused (#8369).
class C:
    v: int = 1


def f() -> int:
    o = c
    return o.v


c = C()
assert f() == 1
