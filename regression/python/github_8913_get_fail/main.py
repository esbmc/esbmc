prec = {"-": 1}
ops = []
for t in ["-", "-"]:
    if ops:
        assert prec.get(ops[-1], 0) == 0
    ops.append(t)
