class Leaf:
    def value(self) -> int:
        return 7

class Mid:
    def get_leaf(self) -> Leaf:
        leaf = Leaf()
        return leaf

class Host:
    def make_mid(self) -> Mid:
        mid = Mid()
        return mid

h = Host()
result = h.make_mid().get_leaf().value()
assert result == 99  # wrong value, should be 7
