# `finally` must run on both the normal-completion path and after a caught
# exception. Each block records a distinct contribution so the final assert
# fails if any finally (or handler) were skipped or run twice.
log = 0

# Normal completion: try body then finally.
try:
    log = log + 1
finally:
    log = log + 10

# Caught exception: handler then finally.
try:
    raise ValueError("e")
except ValueError:
    log = log + 100
finally:
    log = log + 1000

assert log == 1111
