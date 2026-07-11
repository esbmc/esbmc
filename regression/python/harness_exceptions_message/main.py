# Verification harness for exception message payloads
# (src/python-frontend/models/exceptions.py).
#
# The message passed to an exception constructor survives to the handler:
# str(e) returns it. This is intrinsic-free and also runs under CPython.
#
# ENSURES:
#   E1: the caught exception carries the message it was raised with [payload
#       survives the raise/catch round-trip]
#   E2: control reaches the handler
caught: bool = False

try:
    raise ValueError("specific-message")
except ValueError as e:
    caught = True
    assert str(e) == "specific-message"      # E1

assert caught                                 # E2
