# `finally` must run even when the exception is not caught by any handler:
# the handler does not match, so control reaches finally, runs it, and then
# the exception re-propagates. The `assert False` inside finally is reachable
# only if finally executes on this uncaught path -- it pins that behaviour.
try:
    raise ValueError("e")
except KeyError:
    pass
finally:
    assert False
