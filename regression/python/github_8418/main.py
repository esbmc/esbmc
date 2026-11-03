def half(v: int) -> float:
    return v / 2


# Both joined values are stored exactly in the double
def scale(c: bool) -> float:
    if c:
        a = 3
    else:
        a = 0.5
    return a * 2


# A float call result widens the int name instead of being truncated
def from_call(b: bool, x: int) -> float:
    if b:
        a = 1
    else:
        a = half(x)
    return a * 2


# A read before the float binding sees only the int
def later(c: bool) -> int:
    a = 1
    r = a + 1
    if c:
        a = 2.5
    return r


# In a loop the later float binding reaches the read
def in_loop(c: bool) -> float:
    a = 1
    s = 0.0
    for i in range(2):
        s = s + a
        if c:
            a = 2.0
    return s


assert scale(True) == 6
assert scale(False) == 1.0
x = nondet_int() % 10
assert from_call(False, x) == x
assert later(nondet_bool()) == 2
c = nondet_bool()
assert in_loop(c) == (3.0 if c else 2.0)
