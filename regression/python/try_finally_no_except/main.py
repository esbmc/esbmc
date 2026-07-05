# Bare `try`/`finally` with no `except`. This used to abort during conversion
# (a cpp-catch needs >= 2 operands); the finally-rethrow catch-all now supplies
# the second operand, and finally runs on the normal-completion path.
x = 0
try:
    x = 1
finally:
    x = x + 10

assert x == 11
