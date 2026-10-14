import lib
import lib as L

g = lib.inc
assert g(1) == 2
assert lib.add(1) == 2
assert lib.add(a=2) == 3
assert L.dec(3) == 2
args = [7, 2]
assert lib.sub(*args) == 5
