class A:
    def make(self):
        return A()


assert isinstance(A().make(), A)
