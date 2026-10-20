def double(f):

    def wrapper(self, x: int) -> int:
        return 2 * f(self, x)

    return wrapper


class C:

    @double
    def g(self, x: int) -> int:
        return x


assert C().g(3) == 6
