class C:
    def one(self):
        return 1

    def get(self):
        return self.one

g = C().get()
