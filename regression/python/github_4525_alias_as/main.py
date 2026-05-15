from lib import nl_affine_range as aff

count: int = 0
for i in aff(5):
    count = count + 1

assert count == 5
