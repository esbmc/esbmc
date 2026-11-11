# A bare `raise` re-raises the exception currently being handled. Here the outer
# `except ValueError` handler fully handles an inner KeyError in a nested
# try/except, then `raise` must re-raise the *outer* ValueError (the one this
# handler is for), not the inner KeyError.
try:
    try:
        raise ValueError()
    except ValueError:
        try:
            raise KeyError()
        except KeyError:
            pass
        raise
except ValueError:
    result = 1
except KeyError:
    result = 2

assert result == 1
