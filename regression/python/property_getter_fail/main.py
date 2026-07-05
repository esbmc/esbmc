# Negative variant of property_getter: the getter runs and returns the real
# computed value, but the assertion does not hold. Confirms the property value
# flows precisely (a wrong claim is refuted, not vacuously accepted).

class Circle:
    def __init__(self, r):
        self._r = r

    @property
    def area(self):
        return self._r * self._r


c = Circle(3.0)
assert c.area == 99.0
