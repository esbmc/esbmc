xs: list[int] = [5]
if xs and (v := xs.pop()):
    assert v == 4
