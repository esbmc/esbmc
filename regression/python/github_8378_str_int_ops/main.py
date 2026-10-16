x = "ab"

caught = 0
try:
    y = x - 1
except TypeError:
    caught += 1
try:
    y = x / 1
except TypeError:
    caught += 1
try:
    y = x // 1
except TypeError:
    caught += 1
try:
    y = x ** 1
except TypeError:
    caught += 1
try:
    y = x < 1
except TypeError:
    caught += 1
try:
    y = x <= 1
except TypeError:
    caught += 1
try:
    y = x > 1
except TypeError:
    caught += 1
try:
    y = x >= 1
except TypeError:
    caught += 1
try:
    y = -x
except TypeError:
    caught += 1

assert caught == 9
