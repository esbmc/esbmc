# #10143: the value of a str-and-bool and/or is the str or the bool.
z = "" and True
assert not z
assert z == ""

w = True and ""
assert not w

s = ""
v = s or True
assert v
assert v == True

u = "a" and False
assert not u

q = str(5) and True
assert q == True

p = "a" and True and 1
assert p == 1


def f() -> None:
    t = ""
    r = t and True
    assert not r


f()
