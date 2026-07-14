# Verification harness for exception class hierarchy
# (src/python-frontend/models/exceptions.py).
#
# ValueError subclasses Exception, so an `except Exception` handler catches a
# raised ValueError. This is intrinsic-free and also runs under CPython.
#
# ENSURES:
#   E1: an `except Exception` handler catches a raised ValueError [subclass is
#       caught by a base-class handler]
#   E2: control reaches the handler exactly once
caught: bool = False

try:
    raise ValueError("boom")
except Exception:
    caught = True

assert caught  # E1, E2
