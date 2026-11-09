d = {}
d['p'] = ["a"]
d['p'].append("x")
assert len(d['p']) == 2
assert d['p'][1] == "x"
