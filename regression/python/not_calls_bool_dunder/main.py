# `not obj` is a truth test and must go through the class's __bool__, as `if`,
# `while`, a ternary and bool() already did. It used to cast the object
# straight to bool instead, so `not b` was false for an object __bool__ calls
# false.


class Falsy:
    def __bool__(self) -> bool:
        return False


class Truthy:
    def __bool__(self) -> bool:
        return True


f = Falsy()
t = Truthy()

assert not f
assert not not t

# The contexts that already worked must keep working.
r = 1 if f else 2
assert r == 2

n = 0
if t:
    n = 1
assert n == 1

assert bool(f) is False
