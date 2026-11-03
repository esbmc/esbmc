def f(tokens):
    prec = {"-": 1}
    ops = []
    for t in tokens:
        if ops:
            assert prec[ops[-1]] == 1
        ops.append(t)


f(["-", "-"])
