# Arithmetic on None raises TypeError (GitHub #6260).
x = None
caught = False
try:
    y = x + 1
except TypeError:
    caught = True

assert caught
