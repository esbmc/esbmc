# `import math as f` rebinds f, so h() on that arm raises TypeError.
from helper import f, g
import math as f

h = f if nondet_bool() else g
r = h()
