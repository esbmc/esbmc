class NeedsBody:
    def method(self) -> None:
        pass

    @staticmethod
    def another() -> None:
        pass

obj: NeedsBody = NeedsBody()
obj.method()
NeedsBody.another()

