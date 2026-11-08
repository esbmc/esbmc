# Issue #9838: a raising operand of an and/or nested in another operand is
# guarded, so it raises only where Python evaluates it.
class C:
    pass


c = C()
try:
    x = False or (False and c.limit)
    assert False
except AttributeError:
    pass
