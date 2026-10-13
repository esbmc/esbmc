class C:
    @staticmethod
    def ident(x):
        return x

assert C().ident(5) == 5
