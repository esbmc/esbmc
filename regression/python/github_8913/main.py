prec = {"-": 1}
ops = []
for t in ["-", "-"]:
    if ops:
        assert prec[ops[-1]] == 1
    ops.append(t)
