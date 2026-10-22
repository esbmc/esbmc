prec = {"-": 1, "+": 2}
ops = []
for t in ["-", "+"]:
    if ops:
        assert prec[ops[-1]] == 2
    ops.append(t)
