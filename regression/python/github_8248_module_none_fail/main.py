def scale(flag: bool) -> float:
    return 2.5 if flag else 0.0


v = None
if scale(False) > 1.0:
    v = 2.5
w = v * 2
