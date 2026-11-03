# github #8942: a method returning a BinOp over self attributes was typed
# float although the attributes are ints.
class R:

    def __init__(self, w, h, k):
        self.w = w
        self.h = h
        self.k = k

    def area(self):
        return self.w * self.w

    def size(self):
        return self.w * self.h + self.h

    def scaled(self):
        return self.w * self.k


class F:

    def __init__(self, x):
        self.x = x

    def double(self):
        return self.x * 2


class G:

    def __init__(self, n: int, b: bool):
        self.n = n
        self.b = b

    def inv(self):
        return self.n**-1

    def inc(self):
        return self.b + 1


class S:

    def __init__(self, s):
        self.s = s
        self.z = 3

    def cube(self):
        return self.z * self.z * self.z

    def twice(self):
        return self.s * 2


r = R(3, 4, 0.5)
a = r.area()
assert isinstance(a, int)
assert a == 9
b = r.size()
assert isinstance(b, int)
assert b == 16
c = r.scaled()
assert isinstance(c, float)
assert c == 1.5
assert F(1.25).double() == 2.5
g = G(2, True)
assert g.inv() == 0.5
assert g.inc() == 2
t = S("ab")
assert t.twice() == "abab"
v = t.cube()
assert isinstance(v, int)
assert v == 27
