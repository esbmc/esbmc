class A:
    def make(self):
        return A()


class B:
    pass


assert isinstance(A().make(), B)
