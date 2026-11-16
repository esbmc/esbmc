class C:
    def m(self, v: bool = False) -> bool:
        return not v

    def run(self) -> bool:
        ok: bool = self.m(**{})
        return ok


assert not C().run()
