add = 1 + 0
sub = 1 - 0
mul = 1 * 0
div = 1 / 1
idiv = -5 // 2
assert (idiv == -3)
idiv = 5 // 2
assert (idiv == 2)
mod = 2 % 2
bitor = 1 | 1
bitand = 1 & 0
bitxor = 3 ^ 1
bitlsh = 2 << 1
bitrsh = 2 >> 1


def add_nums(x, y):
    z = x + y
    return z


assert add_nums(1, 2) == 3
