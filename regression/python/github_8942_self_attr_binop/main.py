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

    def __init__(self, w: int, b: bool):
        self.w = w
        self.b = b

    def inv(self):
        return self.w**-1

    def inc(self):
        return self.b + 1


class S:

    def __init__(self, s):
        self.s = s

    def twice(self):
        return self.s * 2


a = R(3, 4, 0.5).area()
assert isinstance(a, int)
assert a == 9
b = R(3, 4, 0.5).size()
assert isinstance(b, int)
assert b == 16
c = R(3, 4, 0.5).scaled()
assert isinstance(c, float)
assert c == 1.5
assert F(1.25).double() == 2.5
assert G(2, True).inv() == 0.5
assert G(2, True).inc() == 2
assert S("ab").twice() == "abab"
