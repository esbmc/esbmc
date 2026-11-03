class C:
    def m(self, v: int) -> int:
        return v

    def run(self) -> int:
        d = {"v": 1}
        return self.m(**d)


assert C().run() == 1
