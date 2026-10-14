from lib_b import aff

count: int = 0
for i in aff(3):
    count = count + 1

assert count == 3
