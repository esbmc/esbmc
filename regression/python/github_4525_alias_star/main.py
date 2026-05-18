from lib import *

count: int = 0
for i in nl_affine_range(4):
    count = count + 1

assert count == 4
