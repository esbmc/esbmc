# Reading a @property attribute invokes the decorated getter method. Previously
# `obj.prop` raised "Attribute 'prop' not found" because the property method was
# not a struct field. Accessing it now rewrites to the getter call obj.prop().

class Circle:
    def __init__(self, r):
        self._r = r

    @property
    def radius(self):
        return self._r

    @property
    def area(self):
        return self._r * self._r


c = Circle(3.0)
# External access, and a property whose getter computes a value.
assert c.radius == 3.0
assert c.area == 9.0
# Property used in arithmetic.
assert c.area + 1.0 == 10.0


class Sub(Circle):
    pass


# Property inherited from the base class.
s = Sub(4.0)
assert s.radius == 4.0
