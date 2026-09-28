class Box:
    def any(self, value):
        return value == 3


b = Box()
assert b.any(3)

