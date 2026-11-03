def scale(flag: bool) -> float:
    return 2.5 if flag else 0.0


v = None
if scale(True) > 1.0:
    v = 2.5
assert v is not None
w = v * 2
assert w == 5.0
