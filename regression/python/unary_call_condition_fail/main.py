# An if/while test that is -, ~ or + over a call kept only the bare call: the
# operator was dropped (if) or turned into `not` (while).
def h(a: int) -> int:
    return a


if ~h(0):
    assert False
