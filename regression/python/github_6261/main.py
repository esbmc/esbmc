# len() on an object with no __len__ raises TypeError (GitHub #6261).
n = 5
caught = False
try:
    y = len(n)
except TypeError:
    caught = True

assert caught
