# Issue #9811: iterating a list literal holding an `X or None` name.
s = 0 or None
for v in [s]:
    assert v is not None
