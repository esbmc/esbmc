SYSTEM = [([1.0, 2.0], 3.0), ([4.0, 5.0], 6.0)]


def total(bodies=SYSTEM):
    s = 0.0
    for ([x, y], m) in bodies:
        s += x * m
    return s


assert total() == 27.0
