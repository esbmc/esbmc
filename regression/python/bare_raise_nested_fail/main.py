l = [1, 2]
try:
    x = l[5]
except IndexError as e:
    try:
        d = {1: 2}
        y = d[3]
    except KeyError:
        pass
    raise
