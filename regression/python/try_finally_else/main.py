# The `else` clause of a try runs only when the body completes without an
# exception, and its own exceptions are not caught by this try's handlers.
# ESBMC's lowering does not model that, so a non-empty `else` is refused
# rather than silently dropped. (Valid Python; runs cleanly under CPython.)
x = 0
try:
    x = 1
except ValueError:
    x = 2
else:
    x = 3
finally:
    x = 4

print(x)
