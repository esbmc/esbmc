# A type not in the tuple is not caught and propagates as an uncaught exception.
try:
    raise TypeError()
except (ValueError, KeyError):
    pass
