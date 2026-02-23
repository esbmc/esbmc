
def shunting_yard(tokens):
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    rpntokens = []
    opstack = []
    for token in tokens:
        if isinstance(token, int):
            rpntokens.append(token)
        else:
            while opstack and precedence[token] <= precedence[opstack[-1]]:
                rpntokens.append(opstack.pop())
            opstack.append(token)

    while opstack:
        rpntokens.append(opstack.pop())

    return rpntokens

assert shunting_yard([]) == []
assert shunting_yard([30]) == [30]
assert shunting_yard([10, "-", 5, "-", 2]) == [10, 5, "-", 2, "-"]
assert shunting_yard([34, "-", 12, "/", 5]) == [34, 12, 5, "/", "-"]
assert shunting_yard([4, "+", 9, "*", 9, "-", 10, "+", 13]) == [4, 9, 9, "*", "+", 10, "-", 13, "+"]
assert shunting_yard([7, "*", 43, "-", 7, "+", 13, "/", 7]) == [7, 43, "*", 7, "-", 13, 7, "/", "+"]

