class C:
    limit = 1

    def put(self) -> None:
        while len(self) >= self.limit:
            pass


C().put()
