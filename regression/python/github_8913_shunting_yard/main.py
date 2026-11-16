precedence = {"-": 1, "+": 1}


def convert(tokens):
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


assert convert([10, "-", 5, "-", 2]) == [10, 5, "-", 2, "-"]
