from lib import affine_range

count: int = 0
for i in affine_range(5):
    count = count + 1

assert count == 5
