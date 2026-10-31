def f(tokens):
    rpn = []
    ops = []
    for token in tokens:
        ops.append(token)
    while ops:
        rpn.insert(0, ops.pop())
    return rpn


assert f([10, "-", 5]) == [5, "-", 10]
