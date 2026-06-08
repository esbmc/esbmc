class Leaf:
    def value(self) -> int:
        return 7


class Mid:
    def leaf(self) -> Leaf:
        return Leaf()


class Root:
    def mid(self) -> Mid:
        return Mid()


x = Root().mid().leaf().value()

assert x == 7
