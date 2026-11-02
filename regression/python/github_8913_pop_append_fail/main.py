def f(tokens):
    rpn = []
    ops = []
    for token in tokens:
        ops.append(token)
    while ops:
        rpn.append(ops.pop())
    return rpn


assert f([10, "-", 5]) == [10, "-", 5]
