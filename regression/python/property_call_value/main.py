class T:

    @property
    def n(self) -> int:
        return 3

    @property
    def u(self):
        return 2.5

    @property
    def bad(self) -> int:
        raise ValueError("boom")

    def use(self) -> int:
        return self.n()


class S(T):
    pass


def main() -> None:
    t = T()
    caught = 0
    try:
        t.n()
    except TypeError:
        caught += 1
    try:
        x = t.u() + 1
    except TypeError:
        caught += 1
    try:
        t.use()
    except TypeError:
        caught += 1
    try:
        y = S().n()
    except TypeError:
        caught += 1
    try:
        T.n()
    except TypeError:
        caught += 1
    try:
        t.bad()
    except ValueError:
        caught += 10
    assert t.n == 3
    assert caught == 15


main()
