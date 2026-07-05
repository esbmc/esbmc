# Negative variant of explicit_base_init: the base initialiser runs via the
# explicit `Base.__init__(self, ...)` call, but the assertion does not hold.
# Confirms the base initialiser writes to the real instance (so the value is
# decided), not vacuously accepted.

class Base:
    def __init__(self, name):
        self.name = name


class Child(Base):
    def __init__(self, name):
        Base.__init__(self, name)


c = Child("bot")
assert c.name == "WRONG"
