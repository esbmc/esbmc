class Foo:
    pass


class Bar:
    pass


def create(s: int) -> None:
    if s == 1:
        obj = Foo()
    else:
        obj = Bar()
        assert not isinstance(obj, Bar)


create(2)
