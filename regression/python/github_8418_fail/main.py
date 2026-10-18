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


# Each wrong assertion sits on its own path, so each is reported
k = nondet_int()
x = nondet_int() % 10
c = nondet_bool()
if k == 0:
    assert scale(False) == 2.0
elif k == 1:
    assert from_call(False, x) == x + 1
elif k == 2:
    assert later(c) == 3
else:
    assert in_loop(c) == 2.0
