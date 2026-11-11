def f(tokens):
    rpn = []
    ops = []
    for token in tokens:
        ops.append(token)
    while ops:
        rpn.append(ops.pop())
    return len(rpn)


assert f(["+", "-", "*"]) == 3
