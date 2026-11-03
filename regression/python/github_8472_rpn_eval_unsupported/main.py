# Token lists of different shapes, one reached through another list parameter:
# every token is read through its runtime tag (#8472). Arithmetic on a value
# read that way is refused; before, the float tokens were lost and the pops
# reported an IndexError CPython never raises.
def rpn(tokens):
    stack = []
    for t in tokens:
        if isinstance(t, float):
            stack.append(t)
        else:
            a = stack.pop()
            b = stack.pop()
            if t == "+":
                stack.append(b + a)
            else:
                stack.append(b - a)
    return stack.pop()


def first(d):
    return rpn(d[0])


assert first([[3.0, 5.0, "+"]]) == 8.0
assert rpn([2.0, 2.0, "+", 1.0, "-"]) == 3.0
