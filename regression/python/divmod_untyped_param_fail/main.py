def to_xy(pos):
    y, x = divmod(pos, 9)
    return x, y


def split(a, b):
    return divmod(a, b)


assert divmod(10, 9) == (1, 0)
