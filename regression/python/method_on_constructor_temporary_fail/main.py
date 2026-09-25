class C:
    def get(self):
        return 7


# C().get() calls the class's own method, which returns 7.
assert C().get() == 8
