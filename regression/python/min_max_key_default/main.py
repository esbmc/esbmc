e = []
assert min(e, key=lambda x: -x, default=7) == 7
assert max([3, 1, 2], key=lambda x: -x, default=0) == 1
assert max(e, key=abs, default=-1) == -1
