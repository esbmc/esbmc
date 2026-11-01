ops = [1, "a"]
rpn = []
rpn.append(ops.pop())
assert len(ops) == 1
assert len(rpn) == 1
assert rpn[0] == "a"
