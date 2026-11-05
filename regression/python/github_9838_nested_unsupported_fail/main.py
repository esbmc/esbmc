# Issue #9838: a raising operand of an and/or nested in another operand cannot
# be guarded, so it is refused rather than raised where Python skips it.
class C:
    pass


c = C()
try:
    x = False or (False and c.limit)
    assert False
except AttributeError:
    pass
