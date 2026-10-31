class A:
    pass


class B:
    pass


o = B() if nondet_bool() else A()


def is_a(o: A) -> bool:
    return isinstance(o, A)


assert not is_a(A())
