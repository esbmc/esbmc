# github #9629: repr() never calls __str__.
class S:
    def __str__(self):
        return "<S>"


s = S()
assert repr(s) == "<S>"
