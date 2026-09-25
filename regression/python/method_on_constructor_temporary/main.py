# A method called on a constructor temporary. When its name collides with a
# dict or list method the call used to be claimed by the container handler,
# which then failed looking the class up as a dict variable. A non-colliding
# name (value) never came through that path, and a named receiver resolves its
# type, so only this shape was affected.


class C:
    def get(self):
        return 7

    def pop(self):
        return 8

    def update(self):
        return 9

    def clear(self):
        return 10

    def value(self):
        return 11


assert C().get() == 7
assert C().pop() == 8
assert C().update() == 9
assert C().clear() == 10
assert C().value() == 11

# A named receiver keeps working.
c = C()
assert c.get() == 7

# Real containers still reach their own methods.
d = {"a": 1}
assert d.get("a") == 1
assert d.get("z", 9) == 9

v = [1, 2]
assert v.pop() == 2
assert len(v) == 1
