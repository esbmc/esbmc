def f() -> None:
    d = {}
    d['p'] = []
    d['p'].append(3)
    d['p'].append(4)
    assert len(d['p']) == 2
    assert d['p'][1] == 4

f()
