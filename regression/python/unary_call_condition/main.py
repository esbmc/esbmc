# An if/while test that is -, ~ or + over a call kept only the bare call: the
# operator was dropped (if) or turned into `not` (while).
def h(a: int) -> int:
    return a


if ~h(-1):
    assert False
if not h(0):
    x = 1
else:
    assert False
reached = False
while -h(1):
    reached = True
    break
assert reached
