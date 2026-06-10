# A bare `raise` re-raises the exception currently being handled. Here the outer
# `except ValueError` handler fully handles an inner KeyError in a nested
# try/except, then `raise` must re-raise the *outer* ValueError (the one this
# handler is for), not the inner KeyError.
#
# KNOWNBUG: Python's exception lowering currently re-raises a bare `raise` from
# the global exception state (which the completed inner handler overwrote with
# KeyError), so it wrongly re-raises KeyError. A correct model needs a per-thread
# handled-exception stack for Python (the C++ handled-stack OM is not linked for
# Python because it drags the whole std::terminate closure). Once that lands this
# test verifies SUCCESSFUL and can move to CORE.
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
