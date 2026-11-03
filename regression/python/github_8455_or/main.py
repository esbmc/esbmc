xs: list[int] = []
if not xs or (v := xs.pop()) > 0:
    assert len(xs) == 0
