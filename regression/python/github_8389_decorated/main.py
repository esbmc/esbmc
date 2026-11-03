def as_str(fn):

    def inner(v, h):
        return fn(v, str)

    return inner


@as_str
def is_a(v, h) -> bool:
    return isinstance(v, h)


assert is_a("a", int)
