def outer() -> int:

    def times(k: int):

        def dec(f):

            def w(x: int) -> int:
                return k * f(x)

            return w

        return dec

    @times(3)
    def h(x: int) -> int:
        return x + 1

    return h(1)


assert outer() == 6
