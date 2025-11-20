
def rpn_eval(tokens):
    def op(symbol, a, b):
        return {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }[symbol](a, b)

    stack = []

    for token in tokens:
        if isinstance(token, float):
            stack.append(token)
        else:
            a = stack.pop()
            b = stack.pop()
            stack.append(
                op(token, b, a)
            )

    return stack.pop()

"""
def rpn_eval(tokens):
    def op(symbol, a, b):
        return {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }[symbol](b, a)

    stack = Stack()

    for token in tokens:
        if isinstance(token, float):
            stack.push(token)
        else:
            a = stack.pop()
            b = stack.pop()
            stack.push(
                op(token, a, b)
            )

    return stack.pop()
"""

assert rpn_eval([3.0, 5.0, "+", 2.0, "/"]) == 4.0
assert rpn_eval([2.0, 2.0, "+"]) == 4.0
assert rpn_eval([7.0, 4.0, "+", 3.0, "-"]) == 8.0
assert rpn_eval([1.0, 2.0, "*", 3.0, 4.0, "*", "+"]) == 14.0
assert rpn_eval([5.0, 9.0, 2.0, "*", "+"]) == 23.0
assert rpn_eval([5.0, 1.0, 2.0, "+", 4.0, "*", "+", 3.0, "-"]) == 14.0