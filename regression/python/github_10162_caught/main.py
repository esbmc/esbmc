max = 64
caught = False
try:
    max(2, 1)
except TypeError:
    caught = True
assert caught
