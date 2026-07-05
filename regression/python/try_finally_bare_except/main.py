# A bare `except:` catches every exception, so the program completes normally
# and the finally runs after the handler. Regression for a catch_map collision:
# the synthetic finally-rethrow handler must not overwrite the user's bare
# `except:` (both lower to the "ellipsis" exception id).
x = 0
try:
    raise ValueError()
except:
    x = 1
finally:
    x = x + 10

assert x == 11
