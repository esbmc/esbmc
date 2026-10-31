ops = [1, "a"]
rpn = ["x"]
rpn.insert(0, ops.pop())
assert len(ops) == 1
assert len(rpn) == 2
assert rpn[0] == "a"
assert rpn[1] == "x"
