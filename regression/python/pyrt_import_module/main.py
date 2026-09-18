# An imported module is spliced into this one namespace, so a call inside the
# module resolves the same way a call from here does.
import util
from util import twice

assert util.helper(1) == 2
assert util.twice(1) == 3
assert twice(2) == 4
assert util.STEP == 1
