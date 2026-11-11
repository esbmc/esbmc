class C(list):

    def m(self) -> int:
        return len(self)


assert C().m() == 0
