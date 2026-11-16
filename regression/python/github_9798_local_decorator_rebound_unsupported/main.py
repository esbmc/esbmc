def outer() -> int:

    def dec(f):

        def w() -> int:
            return 1 + f()

        return w

    def ident(f):
        return f

    dec = ident

    @dec
    def h() -> int:
        return 1

    return h()


assert outer() == 1
