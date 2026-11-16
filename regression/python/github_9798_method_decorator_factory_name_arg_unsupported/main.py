def times(k):
    def dec(f):
        def w(self):
            return k * f(self)
        return w
    return dec


class A:
    K = 3

    @times(K)
    def m(self):
        return 2


assert A().m() == 6
