# len(self) on a class without __len__ raises TypeError (#9835).
class C:
    limit = 1

    def put(self) -> None:
        while len(self) >= self.limit:
            pass


C().put()
