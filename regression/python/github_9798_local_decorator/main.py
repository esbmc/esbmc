def outer(k: int) -> int:

    def double(f):

        def w(x: int) -> int:
            return 2 * f(x)

        return w

    def add_k(f):

        def w(x: int) -> int:
            return k + f(x)

        return w

    @double
    def h(x: int) -> int:
        return x + 1

    @add_k
    @double
    def g(x: int) -> int:
        return x

    return h(3) + g(1)


assert outer(1) == 11
assert outer(0) == 10
