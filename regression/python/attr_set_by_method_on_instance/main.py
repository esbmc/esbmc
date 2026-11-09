class R:

    def link(self) -> None:
        self.prev = 1


a = R()
b = R()
a.prev = 2
b.link()
assert b.prev == 1
