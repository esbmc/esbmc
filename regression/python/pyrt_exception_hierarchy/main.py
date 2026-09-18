# A user exception reaches Exception and BaseException through tp_base, so
# isinstance answers the same questions CPython does.
class Outer(Exception):
    pass


class Inner(Outer):
    pass


e = Inner()
assert isinstance(e, Inner)
assert isinstance(e, Outer)
assert isinstance(e, Exception)
assert isinstance(e, BaseException)
assert not isinstance(e, int)
assert type(e) is Inner
