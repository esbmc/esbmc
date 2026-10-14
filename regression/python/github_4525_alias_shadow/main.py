from lib import nl_affine_range


def nl_affine_range(n):
    return [0, 1]


count: int = 0
for i in nl_affine_range(5):
    count = count + 1

assert count == 2
