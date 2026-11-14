xs = [1, 2]
n = 0
for i in range(2):
    try:
        try:
            if i == 0:
                y = xs[5]
            else:
                z = {1: 2}[3]
        except LookupError:
            try:
                w = {1: 2}[4]
            except LookupError:
                pass
            raise
    except IndexError:
        n = n + 1
    except KeyError:
        n = n + 10
assert n == 11
