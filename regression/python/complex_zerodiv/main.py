z: complex = 1 + 2j
w: complex = 0 + 0j

caught: bool = False
try:
    result: complex = z / w
except ZeroDivisionError:
    caught = True

assert caught

# Mixed complex/real: dividing complex by real 0.0
caught2: bool = False
try:
    result2: complex = z / 0.0
except ZeroDivisionError:
    caught2 = True

assert caught2
