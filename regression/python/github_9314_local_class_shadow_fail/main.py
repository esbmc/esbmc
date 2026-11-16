from mod import make


def f() -> int:
    class C:
        def get(self) -> int:
            return 2

    c = C()
    return c.get()


assert f() == 1
