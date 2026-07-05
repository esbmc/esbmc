# An exception raised in the else clause is not caught by the try's own
# handlers; it propagates as an uncaught exception.
try:
    pass
except ValueError:
    pass
else:
    raise KeyError()
