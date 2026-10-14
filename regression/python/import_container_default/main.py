# Each module is preprocessed on its own, so a call into an imported module
# used to be converted without the arguments that module's defaults supply.
# The omitted container default now reaches the callee as a real list.
from helper import sizes, total

assert sizes() == 0
assert sizes([1, 2, 3]) == 3
assert total() == 0
assert total([1, 2, 3]) == 6
assert total([1, 2], 10) == 13
